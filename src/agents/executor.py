# src/agents/executor.py
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Dict

try:
    import resource
except ImportError:
    resource = None  # Windows does not provide POSIX resource limits

try:
    from src.state import AgentState
except ImportError:
    from state import AgentState

EXECUTION_TIMEOUT_SECONDS = 10
MAX_MEMORY_BYTES = 512 * 1024 * 1024  # 512 MB
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB

# Pure analytical & standard utility modules safe for BI computation.
ALLOWED_MODULES = frozenset({
    "collections",
    "csv",
    "datetime",
    "decimal",
    "functools",
    "io",
    "itertools",
    "json",
    "math",
    "numpy",
    "pandas",
    "re",
    "statistics",
    "typing",
})

# Builtins and introspection attributes commonly abused in Python sandbox escapes.
FORBIDDEN_NAMES = frozenset({
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "quit",
    "setattr",
    "vars",
})

FORBIDDEN_ATTRS = frozenset({
    "ag_frame",
    "cr_frame",
    "f_back",
    "f_builtins",
    "f_code",
    "f_globals",
    "f_locals",
    "func_code",
    "func_globals",
    "gi_code",
    "gi_frame",
})

# Executed inside `python -I` (isolated mode) with a scrubbed environment.
_RUNNER_SOURCE = r"""
import contextlib
import io
import json
import sys
import traceback

ALLOWED_MODULES = frozenset(json.loads(sys.argv[1]))

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise ImportError("Relative imports are blocked in the sandbox.")
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"Import of '{name}' is blocked by sandbox security policy.")
    return __import__(name, globals, locals, fromlist, level)

SAFE_BUILTINS = {
    "__import__": _safe_import,
    "Exception": Exception,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

def main():
    payload = json.loads(sys.stdin.read())
    code = payload["code"]
    # Single namespace dict ensures module-level imports are visible inside defined functions.
    sandbox_ns = {"__builtins__": SAFE_BUILTINS}
    captured = io.StringIO()

    try:
        compiled = compile(code, "<sandbox>", "exec")
        with contextlib.redirect_stdout(captured):
            exec(compiled, sandbox_ns)
            if "execute_analysis" in sandbox_ns and callable(sandbox_ns["execute_analysis"]):
                result = sandbox_ns["execute_analysis"]()
            else:
                result = sandbox_ns.get("result", "Execution succeeded with no explicit return value.")

        output = {
            "ok": True,
            "result": str(result),
            "stdout": captured.getvalue(),
        }
    except Exception:
        output = {
            "ok": False,
            "error": traceback.format_exc(),
        }

    sys.stdout.write(json.dumps(output))

if __name__ == "__main__":
    main()
"""


class SandboxViolationError(Exception):
    """Raised when generated code fails static AST security validation."""


def extract_python_code(text: str) -> str:
    """Extracts python code blocks from markdown responses."""
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return text.strip()


def validate_ast(code: str) -> None:
    """Rejects unsafe imports, builtins, and introspection attributes before execution."""
    tree = ast.parse(code, filename="<sandbox>", mode="exec")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    raise SandboxViolationError(
                        f"Blocked import '{alias.name}': only analytical stdlib/data modules are permitted."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                raise SandboxViolationError("Relative imports are not permitted in the sandbox.")
            root = node.module.split(".")[0]
            if root not in ALLOWED_MODULES:
                raise SandboxViolationError(
                    f"Blocked import from '{node.module}': only analytical stdlib/data modules are permitted."
                )
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                raise SandboxViolationError(f"Blocked builtin or identifier '{node.id}'.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRS:
                raise SandboxViolationError(
                    f"Blocked attribute access '.{node.attr}' (dunder/frame introspection is disabled)."
                )


def _build_scrubbed_env() -> Dict[str, str]:
    """Constructs a minimal process environment with all AWS and Supabase secrets stripped."""
    if os.name == "nt":
        # Windows requires SystemRoot/TEMP for python.exe DLL initialization.
        allowed = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP")
        return {k: os.environ[k] for k in allowed if k in os.environ}
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
    }


def _apply_posix_limits() -> None:
    """Applies CPU, virtual memory, and file-write caps inside the Linux child process."""
    if resource is None:
        return
    resource.setrlimit(resource.RLIMIT_CPU, (EXECUTION_TIMEOUT_SECONDS, EXECUTION_TIMEOUT_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_BYTES))


def run_sandboxed_code(code: str) -> Dict[str, Any]:
    """
    Validates AST and runs generated code in an isolated child interpreter (`python -I`)
    with scrubbed environment variables, temporary cwd, and resource/timeout caps.
    """
    validate_ast(code)

    allowed_arg = json.dumps(sorted(ALLOWED_MODULES))
    payload = json.dumps({"code": code})

    with tempfile.TemporaryDirectory(prefix="bi_sandbox_") as tmp_dir:
        popen_kwargs: Dict[str, Any] = {
            "input": payload,
            "cwd": tmp_dir,
            "env": _build_scrubbed_env(),
            "capture_output": True,
            "text": True,
            "timeout": EXECUTION_TIMEOUT_SECONDS,
        }
        if os.name == "posix" and resource is not None:
            popen_kwargs["preexec_fn"] = _apply_posix_limits

        proc = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER_SOURCE, allowed_arg],
            **popen_kwargs,
        )

    if proc.returncode != 0:
        stderr = proc.stderr.strip() or f"Sandbox process exited with code {proc.returncode}"
        raise RuntimeError(stderr)

    response = json.loads(proc.stdout)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "Unknown sandbox execution error"))
    return response


def code_execution_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes generated Python code in an isolated, credential-free subprocess sandbox.
    Captures execution results or returns error tracebacks for the QA agent to self-heal.
    """
    raw_code = state.get("generated_code", "")
    code = extract_python_code(raw_code)

    try:
        outcome = run_sandboxed_code(code)
        result = outcome["result"]
        return {
            "execution_result": result,
            "final_report": f"BI Analysis Result:\n{result}",
            "error_trace": "",
        }
    except subprocess.TimeoutExpired:
        return {
            "error_trace": f"TimeoutError: Generated code exceeded {EXECUTION_TIMEOUT_SECONDS}s execution limit.",
            "execution_result": "",
        }
    except Exception:
        return {
            "error_trace": traceback.format_exc(),
            "execution_result": "",
        }
