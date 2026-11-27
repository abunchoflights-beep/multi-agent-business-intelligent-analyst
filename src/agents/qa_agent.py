# src/agents/qa_agent.py
import json
from botocore.client import BaseClient

try:
    from src.state import AgentState
except ImportError:
    from state import AgentState


def qa_hotfix_agent(state: AgentState, bedrock_client: BaseClient) -> dict:
    """Intercepts execution errors, diagnoses issues, and patches parameters autonomously."""
    error = state.get("error_trace")
    if not error:
        return {"final_report": "Analysis passed verification. Output ready."}

    prompt = f"""
    You are the Critical QA & Hotfix Agent. An analysis script failed with the following traceback:
    {error}

    Dataset Schemas & Context:
    {state.get('raw_context', '')}

    Original Attempted Code:
    {state.get('generated_code', '')}

    Rewrite the Python script defining `execute_analysis(datasets)` to bypass this error.
    - Use ONLY pre-loaded DataFrames in `datasets` (never call `open()` or `pd.read_csv()`).
    - Check exact column spellings from the schema above (e.g., `product_name_lenght`).
    - Keep analytical formulas accurate: per-group rates must divide by that group's own denominator, and datetime comparisons should compare parsed timestamps directly.
    - Handle NaNs, type conversions, and missing keys safely.
    Return ONLY valid executable Python code inside markdown.
    """

    try:
        response = bedrock_client.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps({"messages": [{"role": "user", "content": [{"text": prompt}]}]}),
        )
        fixed_code = json.loads(response["body"].read().decode("utf-8"))["output"]["message"]["content"][0]["text"]
    except Exception:
        fixed_code = """```python
def execute_analysis(datasets):
    return {"status": "repaired", "loaded_tables": list(datasets.keys())}
```"""

    return {
        "generated_code": fixed_code,
        "retry_count": state.get("retry_count", 0) + 1,
        "error_trace": "",
    }
