"""Run LLM-written CP-SAT code (docs/LLM_ROUTING.md, section 15).

The code must define ``solve(data)``.  It is checked statically (imports from
an allow-list, no file, process or reflection access) and then run in a fresh
``python -I`` process with an empty environment (no API keys), a temporary
working directory, and a wall-clock limit.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ALLOWED_IMPORTS = {"ortools", "math", "itertools", "collections", "heapq", "typing", "functools"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input",
                   "breakpoint", "exit", "quit", "help", "memoryview"}

RUNNER = """
import json, sys
ns = {"__name__": "llm_model"}
exec(compile(open("model.py").read(), "model.py", "exec"), ns)
data = json.load(open("data.json"))
route = ns["solve"](data)
if route is not None:
    route = [int(s) for s in route]
print("ROUTE=" + json.dumps(route))
"""


def check_code(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            mods = []
        for m in mods:
            if m.split(".")[0] not in ALLOWED_IMPORTS:
                return f"import of {m} is not allowed (allowed: {', '.join(sorted(ALLOWED_IMPORTS))})"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"use of {node.id} is not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"access to {node.attr} is not allowed"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "solve" for n in tree.body):
        return "the code must define a top-level function solve(data)"
    return None


def run_code(code: str, data: Dict, timeout_s: float = 90.0) -> Tuple[Optional[List[int]], str, float]:
    """(route or None, error message or '', seconds)."""
    problem = check_code(code)
    if problem:
        return None, problem, 0.0
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "model.py").write_text(code)
        Path(tmp, "data.json").write_text(json.dumps(data))
        Path(tmp, "runner.py").write_text(RUNNER)
        tick = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-I", "runner.py"], cwd=tmp, env={"PATH": "/usr/bin:/bin", "HOME": tmp},
                                  capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None, f"timeout: solve(data) did not return within {timeout_s:.0f} s", timeout_s
        seconds = time.perf_counter() - tick
    if proc.returncode != 0:
        return None, "error:\n" + proc.stderr[-1500:], seconds
    line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith("ROUTE=")), None)
    if line is None:
        return None, "no result printed", seconds
    try:
        return json.loads(line[len("ROUTE="):]), "", seconds
    except (ValueError, TypeError) as exc:
        return None, f"result is not a list of satellite ids: {exc}", seconds
