"""GeminiLLM must expose the tool-calling interface graph_agent gates node behaviour on.

Regression: GeminiLLM never set `trigger_function_call` or `tools`, which are what
`graph_agent._get_tool_choice_for_node` and `._tools_for_node` check before doing anything.
Both therefore returned None on every Gemini turn, so a graph node's forced `function_call`
and its per-node tool scope were silently inert — the model was merely free to call any
tool it liked. Observed effect: an end_call node spoke its goodbye and only hung up a turn
later, when the model happened to choose the tool on its own.
"""
import pytest

from bolna.llms.gemini_llm import GeminiLLM


def _tool(name, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} description",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": required or [],
            },
        },
    }


def _llm(*names):
    return GeminiLLM(
        model="gemini-3.5-flash-lite",
        llm_key="test-key",
        api_tools={
            "tools": [_tool(n) for n in names],
            "tools_params": {n: {} for n in names},
        },
    )


def test_the_graph_agent_interface_is_present():
    llm = _llm("end_call", "check_slot")
    # graph_agent gates on exactly these; api_params was already set.
    assert llm.trigger_function_call is True
    assert [t["function"]["name"] for t in llm.tools] == ["end_call", "check_slot"]
    assert sorted(llm.api_params) == ["check_slot", "end_call"]


def test_no_tools_leaves_function_calling_off():
    llm = GeminiLLM(model="gemini-3.5-flash-lite", llm_key="test-key")
    assert llm.trigger_function_call is False
    assert llm.tools == []
    assert llm._build_config("sys").tools is None


def test_every_declaration_is_offered_when_no_subset_is_given():
    config = _llm("end_call", "check_slot")._build_config("sys")
    assert [d.name for d in config.tools[0].function_declarations] == ["end_call", "check_slot"]
    assert config.tool_config is None  # nothing forced


def test_a_node_subset_hides_the_other_tools():
    llm = _llm("end_call", "check_slot")
    config = llm._build_config("sys", tools=[_tool("end_call")])
    assert [d.name for d in config.tools[0].function_declarations] == ["end_call"]
    assert config.tool_config is None


def test_a_forced_tool_becomes_any_mode_restricted_to_that_name():
    from google.genai import types

    llm = _llm("end_call", "check_slot")
    config = llm._build_config(
        "sys", tool_choice={"type": "function", "function": {"name": "end_call"}}
    )
    fcc = config.tool_config.function_calling_config
    assert fcc.mode == types.FunctionCallingConfigMode.ANY
    assert fcc.allowed_function_names == ["end_call"]


@pytest.mark.parametrize("tool_choice", [None, {}, {"function": {}}, {"function": {"name": "nope"}}])
def test_an_unusable_tool_choice_leaves_the_model_free(tool_choice):
    """A name that isn't declared must not pin the model to a tool it cannot call."""
    config = _llm("end_call")._build_config("sys", tool_choice=tool_choice)
    assert config.tool_config is None
