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
    - Return a concise dictionary or list of records answering the question.

    ANALYTICAL ACCURACY RULES:
    - Per-group rates/percentages: When computing a rate per group (such as late-delivery rate by category or seller),
      divide each group's numerator by THAT SAME group's total denominator (e.g. late_orders_in_category / total_orders_in_category * 100),
      NEVER by the global total across all groups unless explicitly asked for "share of total".
    - Also include the group's denominator (`total_orders`) alongside any ranked rate, and apply a sensible minimum volume filter (e.g. >= 30 orders if large dataset) when ranking worst/best rates so 1-order outliers don't skew the top 5.
    - Timestamp comparisons: Parse datetime columns with `pd.to_datetime(..., errors='coerce')` and compare timestamps directly
      (`delivered_dt > estimated_dt`) rather than `.dt.days > 0`, which drops delays under 24 hours.
    - Multi-item orders: When joining `orders` with `order_items`, deduplicate `(order_id, category)` or use `.nunique()` on `order_id` for order-level metrics so multi-item orders are not double-counted.

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
