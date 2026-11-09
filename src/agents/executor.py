# src/agents/executor.py
import re
import traceback
from typing import Dict, Any

try:
    from src.state import AgentState
except ImportError:
    from state import AgentState


def extract_python_code(text: str) -> str:
    """Extracts python code blocks from markdown responses."""
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return text.strip()


def code_execution_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes generated Python code in an isolated dictionary namespace.
    Captures execution results or logs error stack traces for the QA agent to self-heal.
    """
    raw_code = state.get("generated_code", "")
    code = extract_python_code(raw_code)

    local_namespace: Dict[str, Any] = {}
    try:
        # Execute the script definitions (like functions or data pipelines)
        exec(code, {}, local_namespace)

        # Call execute_analysis() if the analyst defined it
        if "execute_analysis" in local_namespace and callable(local_namespace["execute_analysis"]):
            result = local_namespace["execute_analysis"]()
        else:
            result = local_namespace.get("result", "Execution succeeded with no explicit return value.")

        return {
            "execution_result": str(result),
            "final_report": f"BI Analysis Result:\n{result}",
            "error_trace": ""
        }
    except Exception:
        # Intercept the full exception traceback for the QA Hotfix Agent
        return {
            "error_trace": traceback.format_exc(),
            "execution_result": ""
        }

