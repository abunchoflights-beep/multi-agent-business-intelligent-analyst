from typing import TypedDict, List


class AgentState(TypedDict):
    session_id: str
    user_query: str
    file_path: str
    route: str
    selected_tables: List[str]
    data_dir: str
    raw_context: str
    generated_code: str
    execution_result: str
    final_report: str
    error_trace: str
    retry_count: int
