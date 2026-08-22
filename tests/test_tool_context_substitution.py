"""
A tool's $var markers resolve against the call's variables, not just the model's
arguments.

Before this, `prepare_api_request` was handed only the model's own function-call
arguments, so a template that referenced a call variable had nowhere to resolve
from. `substitute_var_markers` kept the marker and the tool POSTed
`{"birth_date": {"$var": "dob"}}` — a JSON object where the API expects a string,
which is silently malformed data reaching a real system rather than an error
anyone would notice.

Two things follow, and both are tested here: the markers now resolve from the
call's variables, and a marker that resolves from nothing at all raises instead
of shipping the marker.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bolna.agent_manager.task_manager import TaskManager  # noqa: E402
from bolna.helpers.function_calling_helpers import (  # noqa: E402
    prepare_api_request,
    unresolved_var_markers,
)


CONTEXT = {
    # tier 1 — known when the call was placed
    "recipient_data": {"patient_first_name": "Jane", "call_sid": "CA123"},
    # tier 2 — captured during the call (extracted variables, tool responses)
    "dob": "1985-03-02",
    "patient_id": "P-77",
    "_node_turns": 3,
}


def _values():
    """The flattening TaskManager applies, without building a TaskManager."""
    return TaskManager._tool_context_values(type("T", (), {"context_data": CONTEXT})())


# --- what a tool can resolve against ------------------------------------------

def test_both_tiers_of_call_variables_are_available():
    values = _values()
    assert values["patient_first_name"] == "Jane"   # placed with the call
    assert values["dob"] == "1985-03-02"            # captured during it


def test_engine_bookkeeping_and_call_identifiers_stay_out():
    """`_node_turns` is noise; call_sid/stream_sid are server-owned and must not
    reach an outbound request just because a template names them."""
    values = _values()
    assert "_node_turns" not in values
    assert "call_sid" not in values


def test_call_start_values_win_a_name_clash():
    """The top-level namespace is uncontrolled — every tool response is merged into
    it — so a stray response key must not redefine what the caller supplied."""
    ctx = {"recipient_data": {"source": "dialer"}, "source": "some-api-response"}
    values = TaskManager._tool_context_values(type("T", (), {"context_data": ctx})())
    assert values["source"] == "dialer"


# --- substitution ---------------------------------------------------------------

def test_a_marker_resolves_from_the_call_variables():
    prepared = prepare_api_request(
        {"first_name": {"$var": "patient_first_name"}, "birth_date": {"$var": "dob"}},
        None, None, context_values=_values(),
    )
    assert prepared["api_params"] == {"first_name": "Jane", "birth_date": "1985-03-02"}


def test_the_models_own_argument_wins_over_a_call_variable():
    """Same name, two sources: the argument the model produced for THIS call is the
    more specific of the two."""
    prepared = prepare_api_request(
        {"dob": {"$var": "dob"}}, None, None,
        context_values=_values(), dob="1990-12-25",
    )
    assert prepared["api_params"] == {"dob": "1990-12-25"}


def test_a_literal_alongside_a_marker_is_sent_verbatim():
    prepared = prepare_api_request(
        {"source": "external", "first_name": {"$var": "patient_first_name"}},
        None, None, context_values=_values(),
    )
    assert prepared["api_params"] == {"source": "external", "first_name": "Jane"}


def test_a_marker_that_resolves_from_nothing_raises():
    """Rather than POST {"$var": "..."} to a real API. trigger_api catches this and
    returns an error body, so the model sees a failed tool call and can route on it."""
    with pytest.raises(ValueError) as e:
        prepare_api_request({"birth_date": {"$var": "dob"}}, None, None, context_values={})
    assert "dob" in str(e.value)


def test_no_context_at_all_still_works_off_the_models_arguments():
    """The pre-existing behaviour, unchanged — every tool that worked before this
    change still resolves the same way."""
    prepared = prepare_api_request({"note": {"$var": "note"}}, None, None, note="hello")
    assert prepared["api_params"] == {"note": "hello"}


def test_nested_and_listed_markers_are_found_when_unresolved():
    assert sorted(unresolved_var_markers(
        {"a": {"b": {"$var": "x"}}, "c": [{"$var": "y"}, "plain"]}
    )) == ["x", "y"]
