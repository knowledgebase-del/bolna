"""
"Store Fields as Variables" — a tool names what its response contributes.

Every tool response used to be merged blind into the top level of context_data,
so captured variables shared one flat namespace across every tool and every
extracted variable, last write wins. Two tools both returning `status` silently
overwrote each other, and a routing condition testing the first read the
second's value.

A tool carrying a `response_variables` mapping now contributes only the values
it names, under the names it gives. Without one, the blind merge is unchanged —
every existing agent keeps working.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bolna.agent_manager.task_manager import TaskManager  # noqa: E402
from bolna.helpers.expression_evaluator import MISSING, resolve_variable  # noqa: E402


def _tm(mapping=None):
    """A stand-in with just the state _captured_response_values reads."""
    params = {"book_appointment": {"url": "https://x"}}
    if mapping is not None:
        params["book_appointment"]["response_variables"] = mapping
    agent = type("T", (), {})()
    agent.kwargs = {"api_tools": {"tools_params": params}}
    return agent


def _capture(mapping, response):
    return TaskManager._captured_response_values(_tm(mapping), "book_appointment", response)


BOOKING = {
    "status": "booked",
    "data": {"appointment": {"id": "APT-88421", "starts": "2026-09-01T14:00:00Z"}},
    "slots": [{"slot_id": "S1"}, {"slot_id": "S2"}],
}


# --- with a mapping ------------------------------------------------------------

def test_only_the_named_values_are_captured():
    captured = _capture({"booking_status": "status"}, BOOKING)
    assert captured == {"booking_status": "booked"}
    assert "data" not in captured and "slots" not in captured


def test_a_value_is_stored_under_the_name_the_author_chose():
    """The whole point: `status` from two different tools can be `verify_status`
    and `booking_status` instead of overwriting each other."""
    assert _capture({"booking_status": "status"}, BOOKING)["booking_status"] == "booked"


def test_a_nested_value_is_reachable_by_dot_path():
    captured = _capture({"confirmation_id": "data.appointment.id"}, BOOKING)
    assert captured == {"confirmation_id": "APT-88421"}


def test_a_list_element_is_reachable_by_index():
    """Real responses put the interesting values inside arrays; a mapping that
    could not index one could only ever name the array itself."""
    captured = _capture({"first_slot": "slots.0.slot_id", "second_slot": "slots.1.slot_id"}, BOOKING)
    assert captured == {"first_slot": "S1", "second_slot": "S2"}


def test_a_path_that_does_not_resolve_is_skipped_not_stored_empty():
    """So an `exists` edge can still tell "the API didn't return it" apart from
    "it came back blank"."""
    captured = _capture({"confirmation_id": "data.appointment.id", "missing": "nope.nothing"}, BOOKING)
    assert captured == {"confirmation_id": "APT-88421"}


def test_an_index_past_the_end_is_skipped():
    assert _capture({"tenth": "slots.9.slot_id"}, BOOKING) == {}


def test_a_falsy_value_is_still_captured():
    """0, "" and False are answers, not absences."""
    assert _capture({"count": "n"}, {"n": 0}) == {"count": 0}
    assert _capture({"flag": "f"}, {"f": False}) == {"flag": False}


def test_malformed_mapping_entries_are_ignored_not_fatal():
    captured = _capture({"": "status", "ok": "status", "bad": None, "empty": ""}, BOOKING)
    assert captured == {"ok": "booked"}


# --- without a mapping: unchanged ------------------------------------------------

def test_no_mapping_merges_the_whole_top_level_as_before():
    assert _capture(None, BOOKING) == BOOKING


def test_an_empty_mapping_is_treated_as_no_mapping():
    assert _capture({}, BOOKING) == BOOKING


def test_a_non_dict_response_contributes_nothing():
    assert _capture(None, "just a string") == {}
    assert _capture(None, ["a", "list"]) == {}


# --- the resolver itself ----------------------------------------------------------

def test_resolve_variable_still_walks_plain_dicts():
    """Expression edges resolve through this same function — extending it for lists
    must not change how a dotted path into nested dicts behaves."""
    assert resolve_variable({"a": {"b": "c"}}, "a.b") == "c"
    assert resolve_variable({"a": {"b": "c"}}, "a.z") is MISSING


def test_resolve_variable_indexes_from_the_end_too():
    assert resolve_variable({"xs": [1, 2, 3]}, "xs.-1") == 3


def test_a_numeric_key_on_a_dict_is_still_a_key():
    """A dict whose keys happen to be digits must not be treated as a list."""
    assert resolve_variable({"xs": {"0": "by-key"}}, "xs.0") == "by-key"
