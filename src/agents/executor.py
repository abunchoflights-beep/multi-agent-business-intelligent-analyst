# src/agents/executor.py
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List

try:
    import pwd
except ImportError:
    pwd = None  # Windows does not provide POSIX pwd module

try:
    from src.data_loader import OLIST_TABLES
    from src.state import AgentState
except ImportError:
    from data_loader import OLIST_TABLES
    from state import AgentState

EXECUTION_TIMEOUT_SECONDS = 15
MAX_MEMORY_BYTES = 1024 * 1024 * 1024  # 1 GB virtual address space (safe for pandas + OpenBLAS)
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_OUTPUT_CHARS = 12000

SANDBOX_USER = os.getenv("SANDBOX_USER", "bi-sandbox")
SANDBOX_PYTHON = os.getenv("SANDBOX_PYTHON", "/opt/bi-sandbox-venv/bin/python")

ALLOWED_CSV_FILENAMES = frozenset(meta["filename"] for meta in OLIST_TABLES.values())

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
    # Block library-level file/network I/O methods; data is injected via `datasets`.
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_table",
    "read_xml",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_html",
    "to_parquet",
    "to_pickle",
    "to_sql",
})

# Executed inside `python -I` (isolated mode), optionally as unprivileged OS user `bi-sandbox`.
_RUNNER_SOURCE = r"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import contextlib
import csv
import inspect
import io
import json
import sys
import traceback

try:
    import resource
except ImportError:
    resource = None

try:
    import pandas as pd
except ImportError:
    pd = None

ALLOWED_MODULES = frozenset(json.loads(sys.argv[1]))
ALLOWED_FILENAMES = frozenset(json.loads(sys.argv[2]))

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

def _load_datasets(table_files):
    datasets = {}
    for table_name, file_path in table_files.items():
        if os.path.basename(file_path) not in ALLOWED_FILENAMES:
            raise PermissionError(f"Blocked unapproved file path: {file_path}")
        if pd is not None:
            datasets[table_name] = pd.read_csv(file_path, encoding="utf-8-sig")
        else:
            with open(file_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
                datasets[table_name] = list(csv.DictReader(f))
    return datasets

def _apply_limits(limits):
    if resource is None:
        return
    cpu = int(limits["cpu_seconds"])
    mem = int(limits["memory_bytes"])
    fsize = int(limits["file_bytes"])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))

def _format_result(value):
    if pd is not None and isinstance(value, (pd.DataFrame, pd.Series)):
        return value.to_string(max_rows=50)
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, ensure_ascii=False, indent=2)
    return str(value)

def main():
    payload = json.loads(sys.stdin.read())
    code = payload["code"]
    table_files = payload.get("table_files", {})
    limits = payload.get("limits", {})

    try:
        datasets = _load_datasets(table_files)
        if limits:
            _apply_limits(limits)

        # Single namespace dict ensures module-level imports are visible inside defined functions.
        sandbox_ns = {
            "__builtins__": SAFE_BUILTINS,
            "datasets": datasets,
        }
        captured = io.StringIO()

        compiled = compile(code, "<sandbox>", "exec")
        with contextlib.redirect_stdout(captured):
            exec(compiled, sandbox_ns)
            fn = sandbox_ns.get("execute_analysis")
            if callable(fn):
                params = inspect.signature(fn).parameters
                result = fn(datasets) if len(params) >= 1 else fn()
            else:
                result = sandbox_ns.get("result", "Execution succeeded with no explicit return value.")

        output = {
            "ok": True,
            "result": _format_result(result)[:12000],
            "stdout": captured.getvalue()[:4000],
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
    """Rejects unsafe imports, builtins, and introspection/I-O attributes before execution."""
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
                raise SandboxViolationError(
                    f"Blocked builtin or identifier '{node.id}'. Use pre-loaded `datasets` instead of file I/O."
                )
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRS:
                raise SandboxViolationError(
                    f"Blocked attribute access '.{node.attr}': use pre-loaded `datasets` DataFrames directly."
                )


def _should_use_os_sandbox_user() -> bool:
    """Detects whether the dedicated unprivileged `bi-sandbox` OS user & venv exist on EC2."""
    if os.name != "posix" or pwd is None:
        return False
    if os.getenv("DISABLE_OS_SANDBOX_USER", "").lower() in ("1", "true", "yes"):
        return False
    if not os.path.exists(SANDBOX_PYTHON):
        return False
    try:
        entry = pwd.getpwnam(SANDBOX_USER)
        return os.geteuid() != entry.pw_uid
    except KeyError:
        return False


def get_sandbox_mode() -> str:
    """Reports active isolation tier for health checks and observability."""
    if _should_use_os_sandbox_user():
        return f"os_user:{SANDBOX_USER}+ast+rlimit"
    return "subprocess:isolated+ast"


def _build_scrubbed_env() -> Dict[str, str]:
    """Constructs a minimal process environment with all AWS and Supabase secrets stripped."""
    if os.name == "nt":
        allowed = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP")
        return {k: os.environ[k] for k in allowed if k in os.environ}
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
    }


def _resolve_table_files(data_dir: str, selected_tables: List[str]) -> Dict[str, str]:
    """Maps selected table names to existing CSV paths inside `data_dir`."""
    if not data_dir or not selected_tables:
        return {}
    base = Path(data_dir)
    resolved: Dict[str, str] = {}
    for table in selected_tables:
        meta = OLIST_TABLES.get(table)
        if not meta:
            continue
        candidate = base / meta["filename"]
        if candidate.exists():
            resolved[table] = str(candidate)
    return resolved


def run_sandboxed_code(
    code: str,
    data_dir: str = "",
    selected_tables: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Validates AST and runs generated code in an isolated child interpreter (`python -I`),
    optionally as unprivileged Linux user `bi-sandbox` with network egress blocked via iptables.
    """
    validate_ast(code)

    allowed_arg = json.dumps(sorted(ALLOWED_MODULES))
    filenames_arg = json.dumps(sorted(ALLOWED_CSV_FILENAMES))
    table_files = _resolve_table_files(data_dir, selected_tables or [])

    payload = json.dumps({
        "code": code,
        "table_files": table_files,
        "limits": {
            "cpu_seconds": EXECUTION_TIMEOUT_SECONDS,
            "memory_bytes": MAX_MEMORY_BYTES,
            "file_bytes": MAX_FILE_SIZE_BYTES,
        },
    })

    if _should_use_os_sandbox_user():
        cmd = [
            "sudo", "-n", "-u", SANDBOX_USER,
            SANDBOX_PYTHON, "-I", "-c", _RUNNER_SOURCE, allowed_arg, filenames_arg,
        ]
    else:
        cmd = [
            sys.executable, "-I", "-c", _RUNNER_SOURCE, allowed_arg, filenames_arg,
        ]

    with tempfile.TemporaryDirectory(prefix="bi_sandbox_") as tmp_dir:
        if os.name == "posix":
            os.chmod(tmp_dir, 0o755)
        proc = subprocess.run(
            cmd,
            input=payload,
            cwd=tmp_dir,
            env=_build_scrubbed_env(),
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT_SECONDS,
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
        outcome = run_sandboxed_code(
            code,
            data_dir=state.get("data_dir", ""),
            selected_tables=state.get("selected_tables", []),
        )
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
