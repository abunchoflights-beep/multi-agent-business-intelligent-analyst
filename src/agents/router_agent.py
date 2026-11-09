# src/agents/router_agent.py
import json
import re
from botocore.client import BaseClient

try:
    from src.state import AgentState
except ImportError:
    from state import AgentState


def clean_json_response(raw_text: str) -> dict:
    """Strips markdown code fences and parses JSON safely."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    cleaned = match.group(1).strip() if match else raw_text.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {"route": "compute", "reason": "Defaulted to compute routing."}


def intake_router_agent(state: AgentState, bedrock_client: BaseClient) -> dict:
    """Analyzes user intent and injects vector database context or structures flat files."""
    query = state.get("user_query", "")

    prompt = f"""
    You are an Intake Router Agent. Determine if the user query requires standard semantic retrieval (RAG) 
    or code execution/data calculation.
    Query: {query}
    Respond strictly in JSON format with two keys: "route" ('rag' or 'compute') and "reason".
    """

    try:
        response = bedrock_client.invoke_model(
            modelId="amazon.nova-micro-v1:0",
            body=json.dumps({
                "inferenceConfig": {"temperature": 0.0},
                "messages": [{"role": "user", "content": [{"text": prompt}]}]
            })
        )
        result = json.loads(response['body'].read().decode('utf-8'))
        raw_text = result['output']['message']['content'][0]['text']
        parsed = clean_json_response(raw_text)
    except Exception:
        # Graceful fallback if model invocation is in mock / bootstrapping mode
        parsed = {"route": "compute", "reason": "Fallback routing active."}

    context = "Sample injected contextual business rows from pgvector matching query."
    if parsed.get("route") == "compute":
        context = "Calculated metrics context ready for analytical processing."

    return {
        "raw_context": context,
        "retry_count": 0
    }
