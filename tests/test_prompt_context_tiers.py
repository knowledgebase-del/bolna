"""
Node prompts can name any variable the call has, not only the ones it started with.

`update_prompt_with_context` reads a single key — `recipient_data` — so only
call-start values were ever visible to a prompt. Everything captured DURING the
call lands at the TOP LEVEL of context_data (task_manager merges each tool
response there; graph_agent merges each transition's extracted variables), which
meant a node prompt saying "your confirmation number is {confirmation_id}" spoke
nothing at all. `_prompt_context` now folds both tiers together.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bolna.agent_types.graph_agent import GraphAgent  # noqa: E402
from bolna.helpers.utils import update_prompt_with_context  # noqa: E402


def _ctx(context_data, frozen=None):
    """A stand-in with just the state _prompt_context reads."""
    agent = type("G", (), {})()
    agent.context_data = context_data
    agent._frozen_time_vars = frozen
    return GraphAgent._prompt_context(agent)


def _render(template, context_data):
    return update_prompt_with_context(template, _ctx(context_data))


def test_a_value_captured_during_the_call_renders_in_a_prompt():
    """The reason this change exists: confirmation_id comes back from the booking
    tool mid-call and is merged at the top level."""
    assert _render(
        "Your confirmation number is {confirmation_id}.",
        {"recipient_data": {}, "confirmation_id": "APT-88421"},
    ) == "Your confirmation number is APT-88421."


def test_call_start_values_still_render_exactly_as_before():
    assert _render(
        "Hi {patient_first_name}, calling from {clinic_name}.",
        {"recipient_data": {"patient_first_name": "Jane", "clinic_name": "Riverside"}},
    ) == "Hi Jane, calling from Riverside."


def test_both_tiers_render_in_one_prompt():
    assert _render(
        "{patient_first_name} was born {dob}.",
        {"recipient_data": {"patient_first_name": "Jane"}, "dob": "1985-03-02"},
    ) == "Jane was born 1985-03-02."


def test_a_call_start_value_wins_a_name_clash():
    """Additive by construction: a prompt that resolved before keeps its value even
    if a tool later returns a field with the same name."""
    assert _render(
        "{source}", {"recipient_data": {"source": "dialer"}, "source": "some-api-response"},
    ) == "dialer"


def test_engine_bookkeeping_is_not_exposed_as_a_variable():
    ctx = _ctx({"recipient_data": {}, "_node_turns": 4, "_last_event": "x"})
    assert "_node_turns" not in ctx["recipient_data"]
    assert "_last_event" not in ctx["recipient_data"]


def test_a_variable_the_call_has_not_captured_yet_renders_empty():
    """DictWithMissing, unchanged — a prompt reaching a value before it exists says
    nothing rather than erroring."""
    assert _render("[{confirmation_id}]", {"recipient_data": {}}) == "[]"


def test_no_context_at_all_is_passed_through_untouched():
    assert _ctx(None) is None
    assert _ctx({}) == {}


def test_captured_values_survive_when_recipient_data_is_missing_entirely():
    """A config that never set recipient_data used to make update_prompt_with_context
    blank every variable. The captured tier still has to reach the prompt."""
    assert _render("{confirmation_id}", {"confirmation_id": "APT-1"}) == "APT-1"


def test_frozen_time_vars_still_win_over_recipient_data():
    """The freeze exists so the system prompt stays byte-identical across turns and
    the prompt cache keeps hitting — folding in the captured tier must not undo it."""
    ctx = _ctx(
        {"recipient_data": {"current_time": "09:00:00 AM", "timezone": "UTC"}},
        frozen={"current_time": "08:00:00 AM"},
    )
    assert ctx["recipient_data"]["current_time"] == "08:00:00 AM"
