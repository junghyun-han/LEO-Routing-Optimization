import json
from pathlib import Path

from multisession.agent.runner import evaluate, parse_spec, run, spec_accuracy
from multisession.agent.tools import Workspace, call_tool
from multisession.conditions import Conditions

ROOT = Path(__file__).resolve().parents[1]


class ScriptedModel:
    """Stands in for an LLM: returns a fixed sequence of tool calls, then stops."""
    name = "scripted"

    def __init__(self, calls):
        self.calls, self.turn = calls, 0

    def chat(self, messages, tools=None):
        if self.turn < len(self.calls):
            name, args = self.calls[self.turn]
            self.turn += 1
            msg = {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{self.turn}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        else:
            msg = {"role": "assistant", "content": "done"}
        return msg, {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}


def _scenario():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_agent_benchmark import base_instance, environment
    sc = json.loads((ROOT / "results_multisession/agent/scenarios.json").read_text())[0]
    _, env = environment()
    inst, ll = base_instance(env, sc)
    return sc, inst, ll, env


def test_tools_reject_bad_arguments_with_valid_options():
    sc, inst, ll, env = _scenario()
    ws = Workspace(inst, ll, env)
    bad = call_tool(ws, "set_region_policy", json.dumps({"region": "atlantis", "policy": "avoid"}))
    assert not bad["ok"] and "north_atlantic" in bad["valid"]["regions"]
    assert not call_tool(ws, "set_failed_satellites", json.dumps({"satellite_ids": [99999]}))["ok"]
    assert not call_tool(ws, "no_such_tool", "{}")["ok"]


def test_agent_loop_with_scripted_model_reaches_ground_truth():
    sc, inst, ll, env = _scenario()
    truth = Conditions.from_dict(sc["truth"])
    calls = [("set_failed_satellites", {"satellite_ids": list(truth.failed_satellites)})]
    calls += [("set_region_policy", {"region": r, "policy": "hotspot"}) for r in truth.hotspot_regions]
    if truth.max_new_links is not None:
        calls.append(("set_new_link_limit", {"limit": truth.max_new_links}))
    calls += [("solve", {"method": "fast"}), ("finish", {"explanation": "ok"})]
    out = run(ScriptedModel(calls), sc["text"], Workspace(inst, ll, env), "agent")
    assert out["error"] is None and out["usage"]["tool_errors"] == 0
    assert spec_accuracy(out["spec"], truth)["all_correct"]
    result = evaluate(out["spec"], truth, inst, ll, env)
    assert result["sessions_on_failed_satellite"] == 0 and result["new_link_limit_ok"]


def test_oneshot_parser_tolerates_extra_text():
    spec = parse_spec('Here you go:\n{"failed_satellites": [1, 2], "max_new_links": 0}\nThanks')
    assert spec["failed_satellites"] == [1, 2] and spec["max_new_links"] == 0 and spec["via"] == {}
