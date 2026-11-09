try:
    from src.state import AgentState
except ImportError:
    from state import AgentState
import json

def qa_hotfix_agent(state: AgentState, bedrock_client) -> dict:
    """Intercepts execution errors, diagnoses issues, and patches parameters autonomously."""
    error = state.get("error_trace")
    if not error:
        return {"final_report": "Analysis passed verification. Output ready."}
        
    prompt = f"""
    You are the Critical QA & Hotfix Agent. An analysis script failed with the following traceback:
    {error}
    
    Original Attempted Code:
    {state['generated_code']}
    
    Rewrite the python script entirely to bypass this error. Fix null pointers, data type casting, or missing keys.
    Return ONLY valid executable python code inside markdown.
    """
    
    response = bedrock_client.invoke_model(
        modelId="amazon.nova-lite-v1:0",
        body=json.dumps({"messages": [{"role": "user", "content": [{"text": prompt}]}]})
    )
    fixed_code = json.loads(response['body'].read().decode('utf-8'))['output']['message']['content'][0]['text']
    
    return {
        "generated_code": fixed_code,
        "retry_count": state["retry_count"] + 1,
        "error_trace": ""
    }