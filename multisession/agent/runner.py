"""Run one scenario with one model, in one of two modes.

* ``oneshot``: the model reads the operator's text and the network state and
  answers once with the conditions as JSON (no tools, no feedback).
* ``agent``: the same text and state, plus the tools; the model calls skills,
  sees validation errors and solver/simulator results, and finishes.

Both modes end with a set of conditions.  For a fair comparison those
conditions are always solved by the same fast solver and replayed in the
simulator under the scenario's true world state (``evaluate``).
"""
from __future__ import annotations

import json
import re
import time
from typing import Dict, List

from ..conditions import OBJECTIVES, REGIONS, Conditions, prepare
from ..fast import solve_fast
from ..simulate import replay
from .tools import TOOL_SCHEMAS, Workspace, call_tool, empty_spec, network_state

SYSTEM = (
    "You are the routing agent of a LEO satellite network operations centre. An operator describes the "
    "current situation and conditions in Korean. Translate every condition faithfully: hard conditions "
    "(failures, regions that must not be crossed, required regions, antenna outages, limits, minimums) must "
    "hold; soft ones (hotspots to pass as little as possible, priorities, objective order) shape the "
    "objective. Do not invent conditions the operator did not state."
)

SPEC_FORMAT = (
    "Answer with one JSON object and nothing else:\n"
    '{"failed_satellites": [int], "avoid_regions": [region], "hotspot_regions": [region], '
    '"via": {destination or "all": region}, "antennas": {station: count}, "max_new_links": int or null, '
    '"min_admitted": {destination: count}, "priority": [destination], "objective": [order of '
    + ", ".join(f'"{o}"' for o in OBJECTIVES) + "]}\n"
    f"regions: {', '.join(REGIONS)}. stations: new_york, london, istanbul, hanoi. "
    "destinations: london, istanbul, hanoi. Default objective order: " + ", ".join(OBJECTIVES) + "."
)


def _user_message(text: str, ws: Workspace, mode: str) -> str:
    state = json.dumps(network_state(ws), ensure_ascii=False)
    tail = SPEC_FORMAT if mode == "oneshot" else (
        "Use the tools to set the conditions, call solve to check the result, fix anything that looks wrong, "
        "and call finish.")
    return f"Operator:\n{text}\n\nNetwork state:\n{state}\n\n{tail}"


def parse_spec(text: str) -> Dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    data = json.loads(match.group(0)) if match else {}
    spec = empty_spec()
    for key in spec:
        if key in data and data[key] is not None:
            spec[key] = data[key]
    if "max_new_links" in data:
        spec["max_new_links"] = data["max_new_links"]
    return spec


def run(model, text: str, ws: Workspace, mode: str, max_turns: int = 16) -> Dict:
    messages: List[Dict] = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": _user_message(text, ws, mode)}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "llm_calls": 0, "tool_calls": 0,
             "tool_errors": 0}
    transcript: List[Dict] = []
    error = None
    tick = time.perf_counter()
    try:
        if mode == "oneshot":
            msg, u = model.chat(messages)
            _add(usage, u)
            transcript.append({"assistant": msg["content"]})
            try:
                ws.spec = parse_spec(msg["content"])
            except (json.JSONDecodeError, TypeError) as exc:
                error = f"unparseable answer: {exc}"
        else:
            for _ in range(max_turns):
                msg, u = model.chat(messages, TOOL_SCHEMAS)
                _add(usage, u)
                messages.append(msg)
                if not msg.get("tool_calls"):
                    transcript.append({"assistant": msg["content"]})
                    break
                for call in msg["tool_calls"]:
                    name, args = call["function"]["name"], call["function"]["arguments"]
                    result = call_tool(ws, name, args)
                    usage["tool_calls"] += 1
                    usage["tool_errors"] += int(not result.get("ok", False))
                    transcript.append({"tool": name, "arguments": args, "result_ok": result.get("ok")})
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": json.dumps(result, ensure_ascii=False, default=str)[:4000]})
                if ws.finished:
                    break
    except Exception as exc:  # API or model failure: score what was set so far
        error = f"{type(exc).__name__}: {exc}"
    usage["wall_s"] = time.perf_counter() - tick
    return {"spec": ws.spec, "usage": usage, "error": error, "transcript": transcript,
            "explanation": ws.explanation}


def _add(total: Dict, u: Dict) -> None:
    total["prompt_tokens"] += u["prompt_tokens"]
    total["completion_tokens"] += u["completion_tokens"]
    total["seconds"] += u["seconds"]
    total["llm_calls"] += 1


def spec_accuracy(spec: Dict, truth: Conditions) -> Dict:
    """Field-by-field agreement with the ground-truth conditions (order matters only for priority/objective)."""
    try:
        got = Conditions.from_dict(spec)
    except (ValueError, TypeError, KeyError) as exc:
        return {"valid": False, "error": str(exc)}
    fields = ("failed_satellites", "avoid_regions", "hotspot_regions", "via", "antennas", "max_new_links",
              "min_admitted", "priority", "objective")
    out = {"valid": True}
    for f in fields:
        out[f] = getattr(got, f) == getattr(truth, f)
    out["all_correct"] = all(out[f] for f in fields)
    return out


def evaluate(spec: Dict, truth: Conditions, instance, latlon, env) -> Dict:
    """Solve the model's conditions with the fast solver, replay under the true world state."""
    world = prepare(instance, truth, latlon)
    try:
        own = prepare(instance, Conditions.from_dict(spec), latlon)
    except (ValueError, TypeError, KeyError):
        own = prepare(instance, Conditions(), latlon)  # unusable answer: route as if no conditions
    solution = solve_fast(own)
    return replay(world, solution, *env)
