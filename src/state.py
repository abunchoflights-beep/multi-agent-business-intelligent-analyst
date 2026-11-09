from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    session_id: str
    user_query: str
    file_path: str
    raw_context: str
    generated_code: str
    execution_result: str
    final_report: str
    error_trace: str
    retry_count: int