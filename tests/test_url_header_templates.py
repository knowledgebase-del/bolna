"""
{{name}} in a tool's URL and header values, and a per-tool request timeout.

A request BODY resolves {"$var": "name"} markers — typed JSON substitution, so a
list stays a list. A URL or a header cannot use those: they are strings that need
the value embedded in other text ("Bearer {{token}}",
"/v1/patients/{{patient_id}}/appointments"). Neither was substituted at all, so a
value could only ever reach a tool's body.

Double braces because that is what the portal's URL and header fields already
contain — this is authorable with no UI change.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bolna.helpers import function_calling_helpers as fch  # noqa: E402
from bolna.helpers.function_calling_helpers import (  # noqa: E402
    resolve_timeout,
    substitute_templates,
    trigger_api,
)


VALUES = {"patient_id": "P-77", "access_token": "tok-abc", "q": "a/b?c=d"}


# --- the substitution primitive -------------------------------------------------

def test_a_template_is_replaced_in_a_longer_string():
    out, missing = substitute_templates("Bearer {{access_token}}", VALUES)
    assert out == "Bearer tok-abc" and missing == []


def test_several_templates_in_one_string():
    out, _ = substitute_templates("/v1/{{patient_id}}/x/{{patient_id}}", VALUES)
    assert out == "/v1/P-77/x/P-77"


def test_a_url_value_is_percent_encoded():
    """A value carrying / or ? would otherwise change the SHAPE of the request
    rather than fill a slot in it."""
    out, _ = substitute_templates("https://x/s/{{q}}", VALUES, quote_value=True)
    assert out == "https://x/s/a%2Fb%3Fc%3Dd"


def test_a_header_value_is_not_encoded():
    out, _ = substitute_templates("Bearer {{q}}", VALUES)
    assert out == "Bearer a/b?c=d"


def test_an_unknown_name_is_reported_and_left_in_place():
    out, missing = substitute_templates("{{nope}}", VALUES)
    assert missing == ["nope"] and out == "{{nope}}"


def test_text_with_no_templates_is_untouched():
    out, missing = substitute_templates("https://api.example.com/v1/patient", VALUES)
    assert out == "https://api.example.com/v1/patient" and missing == []


# --- per-tool timeout ------------------------------------------------------------

def test_timeout_defaults_when_the_tool_sets_none():
    assert resolve_timeout(None) == 10


def test_a_tools_timeout_is_converted_from_milliseconds():
    assert resolve_timeout(30000) == 30


def test_an_absurd_timeout_is_clamped():
    """A tool blocking longer than this leaves the caller listening to silence
    while the call clock runs."""
    assert resolve_timeout(10_000_000) == 120


def test_a_nonsense_timeout_falls_back_to_the_default():
    assert resolve_timeout("soon") == 10
    assert resolve_timeout(0) == 10
    assert resolve_timeout(-5) == 10


# --- through trigger_api ----------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def test_the_url_is_substituted_before_the_ssrf_check(monkeypatch):
    """The ORDER is the security property. validate_outbound_url is the SSRF guard;
    running it on the un-substituted string would let a tool response feed a host
    into {{...}} and sail straight past it."""
    seen = {}

    async def fake_validate(url):
        seen["validated"] = url
        raise fch.SSRFError("blocked")           # stop before any real request

    monkeypatch.setattr(fch, "validate_outbound_url", fake_validate)
    _run(trigger_api(
        url="https://api.example.com/v1/patients/{{patient_id}}/appointments",
        method="post", param=None, api_token=None, headers_data=None,
        meta_info={}, run_id=None, return_response_metadata=True,
        context_values=VALUES,
    ))
    assert seen["validated"] == "https://api.example.com/v1/patients/P-77/appointments"


def test_an_unresolved_url_template_fails_the_tool_call(monkeypatch):
    """Rather than requesting a URL with a literal {{...}} in it. trigger_api turns
    the raise into an error body the model can route on."""
    async def fake_validate(url):
        raise AssertionError("must not reach validation")

    monkeypatch.setattr(fch, "validate_outbound_url", fake_validate)
    out = _run(trigger_api(
        url="https://api.example.com/{{nope}}", method="post", param=None, api_token=None,
        headers_data=None, meta_info={}, run_id=None, return_response_metadata=True,
        context_values=VALUES,
    ))
    assert "nope" in out["error"]


def test_an_unresolved_header_template_fails_the_tool_call(monkeypatch):
    async def fake_validate(url):
        raise AssertionError("must not reach validation")

    monkeypatch.setattr(fch, "validate_outbound_url", fake_validate)
    out = _run(trigger_api(
        url="https://api.example.com/x", method="post", param=None, api_token=None,
        headers_data={"Authorization": "Bearer {{missing_token}}"},
        meta_info={}, run_id=None, return_response_metadata=True,
        context_values=VALUES,
    ))
    assert "missing_token" in out["error"]


def test_the_models_own_argument_wins_over_a_call_variable_in_a_url(monkeypatch):
    seen = {}

    async def fake_validate(url):
        seen["validated"] = url
        raise fch.SSRFError("blocked")

    monkeypatch.setattr(fch, "validate_outbound_url", fake_validate)
    _run(trigger_api(
        url="https://api.example.com/{{patient_id}}", method="post", param=None,
        api_token=None, headers_data=None, meta_info={}, run_id=None,
        return_response_metadata=True, context_values=VALUES, patient_id="P-99",
    ))
    assert seen["validated"] == "https://api.example.com/P-99"


def test_a_url_with_no_templates_reaches_validation_unchanged(monkeypatch):
    """Every tool that worked before this change must be byte-identical after it."""
    seen = {}

    async def fake_validate(url):
        seen["validated"] = url
        raise fch.SSRFError("blocked")

    monkeypatch.setattr(fch, "validate_outbound_url", fake_validate)
    _run(trigger_api(
        url="https://api.example.com/v1/patient?interface=openemr", method="post",
        param=None, api_token=None, headers_data={"x-key": "static"},
        meta_info={}, run_id=None, return_response_metadata=True,
    ))
    assert seen["validated"] == "https://api.example.com/v1/patient?interface=openemr"
