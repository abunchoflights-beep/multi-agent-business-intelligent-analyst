try:
    from src.state import AgentState
except ImportError:
    from state import AgentState
import json

def analytical_agent(state: AgentState, bedrock_client) -> dict:
    """Generates structured python/data transformations or queries based on the context."""
    prompt = f"""
    You are a expert Business Intelligence Data Analyst. Write a clean Python function named `execute_analysis` 
    to process the data context and answer the query.
    Context: {state['raw_context']}
    Query: {state['user_query']}
    
    Respond ONLY with executable code block inside markdown. No explanations.
    """
    response = bedrock_client.invoke_model(
        modelId="amazon.nova-lite-v1:0",
        body=json.dumps({"messages": [{"role": "user", "content": [{"text": prompt}]}]})
    )
    res_text = json.loads(response['body'].read().decode('utf-8'))['output']['message']['content'][0]['text']
    
    return {"generated_code": res_text}
