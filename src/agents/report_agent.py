# src/agents/report_agent.py
import json
from botocore.client import BaseClient

try:
    from src.state import AgentState
except ImportError:
    from state import AgentState


def report_synthesis_agent(state: AgentState, bedrock_client: BaseClient) -> dict:
    """Synthesizes sandbox execution metrics and/or pgvector review matches into an executive BI summary."""
    error_trace = state.get("error_trace", "")
    retry_count = state.get("retry_count", 0)
    execution_result = state.get("execution_result", "")
    rag_context = state.get("rag_context", "")

    if error_trace:
        return {
            "final_report": (
                f"Analysis failed after {retry_count} QA hotfix attempts.\n"
                f"Last Error:\n{error_trace}"
            )
        }

    prompt = f"""
    You are an Executive Business Intelligence Advisor.
    Synthesize the verified evidence below into a crisp executive insight answering the user's question.

    Rules:
    - Cite ONLY exact numbers, categories, themes, and review details present in the evidence below. Never invent metrics.
    - If Portuguese customer reviews are provided, summarize the recurring complaint or praise themes in English and cite representative categories/scores.
    - Lead with a 1-sentence direct answer, followed by 2-4 concise bullet points of key takeaways.
    - Keep it under 220 words.

    User Question: {state.get('user_query', '')}
    Route: {state.get('route', 'compute')} (tables: {state.get('selected_tables', [])}, QA retries: {retry_count})

    Verified Execution Output (Quantitative):
    {execution_result or "N/A (pure semantic RAG route)"}

    Semantic Review Matches (Qualitative via Titan V2 + pgvector):
    {rag_context or "N/A"}
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
        parts = []
        if execution_result:
            parts.append(f"BI Analysis Result:\n{execution_result}")
        if rag_context:
            parts.append(rag_context)
        summary = "\n\n".join(parts) if parts else "No output generated."

    return {"final_report": summary}
