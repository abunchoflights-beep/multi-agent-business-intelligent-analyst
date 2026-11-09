# src/graph.py
from functools import partial
from langgraph.graph import StateGraph, END
from botocore.client import BaseClient

try:
    from src.state import AgentState
    from src.agents.router_agent import intake_router_agent
    from src.agents.analyst_agent import analytical_agent
    from src.agents.qa_agent import qa_hotfix_agent
    from src.agents.executor import code_execution_node
except ImportError:
    from state import AgentState
    from agents.router_agent import intake_router_agent
    from agents.analyst_agent import analytical_agent
    from agents.qa_agent import qa_hotfix_agent
    from agents.executor import code_execution_node


def should_continue(state: AgentState) -> str:
    """
    Self-correcting conditional edge logic:
    - If code execution threw an error AND we have not exceeded 3 retries -> route to QA Agent.
    - If execution succeeded OR retries are exhausted -> proceed to END.
    """
    has_error = bool(state.get("error_trace"))
    retry_count = state.get("retry_count", 0)

    if has_error and retry_count < 3:
        return "qa_hotfix"
    return END


def create_bi_analyst_graph(bedrock_client: BaseClient):
    """
    Assembles the cyclic multi-agent graph with AWS Bedrock client curried into each agent node.
    """
    workflow = StateGraph(AgentState)

    # 1. Register Nodes with Bedrock dependency injection
    workflow.add_node("router", partial(intake_router_agent, bedrock_client=bedrock_client))
    workflow.add_node("analyst", partial(analytical_agent, bedrock_client=bedrock_client))
    workflow.add_node("executor", code_execution_node)
    workflow.add_node("qa_hotfix", partial(qa_hotfix_agent, bedrock_client=bedrock_client))

    # 2. Define standard forward flow
    workflow.set_entry_point("router")
    workflow.add_edge("router", "analyst")
    workflow.add_edge("analyst", "executor")

    # 3. Define conditional self-healing loop
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "qa_hotfix": "qa_hotfix",
            END: END
        }
    )

    # 4. Route repaired code back to executor
    workflow.add_edge("qa_hotfix", "executor")

    return workflow.compile()

