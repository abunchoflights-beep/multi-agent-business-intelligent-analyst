# src/graph.py
from functools import partial
from typing import Optional
from botocore.client import BaseClient
from langgraph.graph import END, StateGraph
from supabase import Client

try:
    from src.state import AgentState
    from src.agents.router_agent import intake_router_agent
    from src.agents.analyst_agent import analytical_agent
    from src.agents.qa_agent import qa_hotfix_agent
    from src.agents.executor import code_execution_node
    from src.agents.report_agent import report_synthesis_agent
except ImportError:
    from state import AgentState
    from agents.router_agent import intake_router_agent
    from agents.analyst_agent import analytical_agent
    from agents.qa_agent import qa_hotfix_agent
    from agents.executor import code_execution_node
    from agents.report_agent import report_synthesis_agent


def route_after_intake(state: AgentState) -> str:
    """
    Skips Python code execution when the query is pure semantic RAG and pgvector
    already returned matched reviews; otherwise sends schemas to the Analyst Agent.
    """
    if state.get("route") == "rag" and state.get("rag_context"):
        return "reporter"
    return "analyst"


def should_continue(state: AgentState) -> str:
    """
    Self-correcting conditional edge logic:
    - If code execution threw an error AND we have not exceeded 3 retries -> route to QA Agent.
    - If execution succeeded OR retries are exhausted -> proceed to executive report synthesis.
    """
    has_error = bool(state.get("error_trace"))
    retry_count = state.get("retry_count", 0)

    if has_error and retry_count < 3:
        return "qa_hotfix"
    return "reporter"


def create_bi_analyst_graph(
    bedrock_client: BaseClient,
    s3_client: Optional[BaseClient] = None,
    supabase_client: Optional[Client] = None,
):
    """
    Assembles the multi-agent graph with AWS Bedrock, S3, and Supabase clients curried into trusted nodes.
    """
    workflow = StateGraph(AgentState)

    # 1. Register Nodes with dependency injection (only trusted nodes receive AWS/DB clients)
    workflow.add_node(
        "router",
        partial(
            intake_router_agent,
            bedrock_client=bedrock_client,
            s3_client=s3_client,
            supabase_client=supabase_client,
        ),
    )
    workflow.add_node("analyst", partial(analytical_agent, bedrock_client=bedrock_client))
    workflow.add_node("executor", code_execution_node)
    workflow.add_node("qa_hotfix", partial(qa_hotfix_agent, bedrock_client=bedrock_client))
    workflow.add_node("reporter", partial(report_synthesis_agent, bedrock_client=bedrock_client))

    # 2. Entry & intake routing (pure RAG -> reporter; compute/hybrid -> analyst -> executor)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_after_intake,
        {
            "reporter": "reporter",
            "analyst": "analyst",
        },
    )
    workflow.add_edge("analyst", "executor")

    # 3. Conditional self-healing loop -> executive report synthesis
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "qa_hotfix": "qa_hotfix",
            "reporter": "reporter",
        },
    )

    # 4. Route repaired code back to executor, and finish after synthesis
    workflow.add_edge("qa_hotfix", "executor")
    workflow.add_edge("reporter", END)

    return workflow.compile()
