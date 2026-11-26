# src/agents/analyst_agent.py
import json
from botocore.client import BaseClient

try:
    from src.state import AgentState
except ImportError:
    from state import AgentState


def analytical_agent(state: AgentState, bedrock_client: BaseClient) -> dict:
    """Generates Pandas analysis code against pre-loaded Olist DataFrames in `datasets`."""
    prompt = f"""
    You are an expert Business Intelligence Data Analyst working with the Olist Brazilian E-Commerce dataset.
    Write a clean Python function named `execute_analysis(datasets)` to answer the user query.

    CRITICAL SANDBOX RULES:
    - All requested CSV tables are ALREADY loaded as pandas DataFrames inside the `datasets` dictionary
      (for example: `datasets["orders"]`, `datasets["order_items"]`, `datasets["products"]`).
    - Do NOT call `open()`, `pd.read_csv()`, `os`, `sys`, or any network/file I/O functions.
    - Only import standard analytical libraries (`pandas`, `numpy`, `math`, `statistics`, `datetime`, `json`, `re`).
    - Convert numpy/pandas scalar values to native Python `int`, `float`, `str`, or `dict` so the result serializes cleanly.
    - Return a concise dictionary or formatted summary answering the question.

    Dataset Schemas & Context:
    {state.get('raw_context', '')}

    User Query: {state.get('user_query', '')}

    Respond ONLY with an executable Python code block inside markdown. No explanations.
    """

    try:
        response = bedrock_client.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps({"messages": [{"role": "user", "content": [{"text": prompt}]}]}),
        )
        res_text = json.loads(response["body"].read().decode("utf-8"))["output"]["message"]["content"][0]["text"]
    except Exception:
        res_text = f"""```python
def execute_analysis(datasets):
    return {{
        "status": "fallback",
        "loaded_tables": list(datasets.keys()),
        "query": {json.dumps(state.get('user_query', ''))}
    }}
```"""

    return {"generated_code": res_text}
