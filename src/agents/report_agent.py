# src/agents/report_agent.py
import json
from botocore.client import BaseClient

try:
    from src.state import AgentState
except ImportError:
    from state import AgentState


def report_synthesis_agent(state: AgentState, bedrock_client: BaseClient) -> dict:
    """Synthesizes raw sandbox execution output into a concise executive BI summary."""
    error_trace = state.get("error_trace", "")
    retry_count = state.get("retry_count", 0)
    execution_result = state.get("execution_result", "")

    if error_trace:
        return {
            "final_report": (
                f"Analysis failed after {retry_count} QA hotfix attempts.\n"
                f"Last Error:\n{error_trace}"
            )
        }

    prompt = f"""
    You are an Executive Business Intelligence Advisor.
    Summarize the verified analytical output below into a crisp executive insight for the user.

    Rules:
    - Cite ONLY exact numbers, categories, and facts present in the Verified Execution Output. Never invent metrics.
    - Lead with a 1-sentence direct answer, followed by 2-4 concise bullet points of key takeaways.
    - Keep it under 180 words.

    User Question: {state.get('user_query', '')}
    Routing Context: {state.get('route', 'compute')} (tables: {state.get('selected_tables', [])}, QA retries: {retry_count})

    Verified Execution Output:
    {execution_result}
    """

    try:
        response = bedrock_client.invoke_model(
            modelId="amazon.nova-micro-v1:0",
            body=json.dumps({
                "inferenceConfig": {"temperature": 0.1},
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
            }),
        )
        summary = json.loads(response["body"].read().decode("utf-8"))["output"]["message"]["content"][0]["text"].strip()
    except Exception:
        summary = f"BI Analysis Result:\n{execution_result}"

    return {"final_report": summary}
