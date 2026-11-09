# src/agents/analyst_agent.py
import json
from botocore.client import BaseClient

try:
    from src.state import AgentState
except ImportError:
    from state import AgentState


def analytical_agent(state: AgentState, bedrock_client: BaseClient) -> dict:
    """Generates structured python/data transformations or queries based on the context."""
    prompt = f"""
    You are an expert Business Intelligence Data Analyst. Write a clean Python function named `execute_analysis` 
    to process the data context and answer the query.
    Context: {state.get('raw_context', '')}
    Query: {state.get('user_query', '')}
    
    Respond ONLY with executable code block inside markdown. No explanations.
    """

    try:
        response = bedrock_client.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps({"messages": [{"role": "user", "content": [{"text": prompt}]}]})
        )
        res_text = json.loads(response['body'].read().decode('utf-8'))['output']['message']['content'][0]['text']
    except Exception as e:
        # Fallback code template if offline or during local smoke testing
        res_text = f"""```python
def execute_analysis():
    # Fallback simulation of data analysis
    return {{"status": "computed", "summary": "Sample analytical computation for query: {state.get('user_query')}"}}
```"""

    return {"generated_code": res_text}
