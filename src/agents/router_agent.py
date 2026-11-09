import json
try:
    from src.state import AgentState
except ImportError:
    from state import AgentState
from botocore.client import BaseClient

def intake_router_agent(state: AgentState, bedrock_client: BaseClient) -> dict:
    """Analyzes user intent and injects vector database context or structures flat files."""
    query = state["user_query"]
    
    # Define tool schema for routing
    prompt = f"""
    You are an Intake Router Agent. Determine if the user query requires standard semantic retrieval (RAG) 
    or code execution/data calculation.
    Query: {query}
    Respond strictly in JSON format with two keys: "route" ('rag' or 'compute') and "reason".
    """
    
    response = bedrock_client.invoke_model(
        modelId="amazon.nova-micro-v1:0",
        body=json.dumps({"inferenceConfig": {"temperature": 0.0}, "messages": [{"role": "user", "content": [{"text": prompt}]}]})
    )
    
    result = json.loads(response['body'].read().decode('utf-8'))
    content = json.loads(result['output']['message']['content'][0]['text'])
    
    # Placeholder for database vector similarity call if route == 'rag'
    # Fallback to pure document loading if specific file_path is active
    return {
        "raw_context": "Sample injected contextual business rows from pgvector matching query.",
        "retry_count": 0
    }
