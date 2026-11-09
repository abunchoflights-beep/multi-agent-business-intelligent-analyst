# src/app.py
import os
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from supabase import create_client, Client

try:
    from src.graph import create_bi_analyst_graph
except ImportError:
    from graph import create_bi_analyst_graph

# Load ONLY non-AWS 3rd-party secrets (like Supabase) from .env if present
load_dotenv()

app = FastAPI(
    title="Multi-Agent BI Analyst",
    description="Production-grade cloud-native Multi-Agent BI Analyst on AWS Bedrock & EC2",
    version="1.0.0"
)

# Notice: NO aws_access_key_id or aws_secret_access_key passed here!
# Only the region is required. Boto3 automatically resolves credentials via:
# 1. ~/.aws/credentials (local development)
# 2. EC2 IAM Instance Profile via IMDSv2 (production EC2)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION
)

s3_client = boto3.client(
    service_name="s3",
    region_name=AWS_REGION
)

# Supabase uses standard env vars (or AWS Systems Manager Parameter Store)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase: Client | None = None
if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
    except Exception:
        supabase = None

# Initialize the self-correcting LangGraph multi-agent workflow
bi_analyst_graph = create_bi_analyst_graph(bedrock_runtime)


class AnalysisRequest(BaseModel):
    user_query: str = Field(
        ...,
        description="The business question or analytical request",
        example="Calculate total quarterly revenue growth and summarize trends"
    )
    file_path: str = Field(
        default="",
        description="Optional S3 URI or file path to analyze",
        example="s3://my-bi-bucket/sales_q3.csv"
    )
    session_id: str = Field(
        default="session-001",
        description="Session identifier for multi-turn state tracking"
    )


class AnalysisResponse(BaseModel):
    session_id: str
    user_query: str
    execution_result: str
    final_report: str
    generated_code: str
    retry_count: int
    error_trace: str


@app.get("/health")
def health_check():
    """Liveness probe for Nginx, load balancers, and deployment verification."""
    return {
        "status": "healthy",
        "service": "bi-analyst-backend",
        "region": AWS_REGION,
        "supabase_configured": supabase is not None
    }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_data(request: AnalysisRequest):
    """
    Executes the multi-agent BI workflow:
    1. Router Agent: Classifies intent & retrieves relevant context.
    2. Analyst Agent: Generates Python data transformation code.
    3. Sandbox Executor: Runs the Python script in an isolated namespace.
    4. QA Hotfix Loop: Catches any runtime exceptions and heals the code.
    """
    initial_state = {
        "session_id": request.session_id,
        "user_query": request.user_query,
        "file_path": request.file_path,
        "raw_context": "",
        "generated_code": "",
        "execution_result": "",
        "final_report": "",
        "error_trace": "",
        "retry_count": 0
    }

    final_state = bi_analyst_graph.invoke(initial_state)

    return AnalysisResponse(
        session_id=final_state.get("session_id", request.session_id),
        user_query=final_state.get("user_query", request.user_query),
        execution_result=final_state.get("execution_result", ""),
        final_report=final_state.get("final_report", ""),
        generated_code=final_state.get("generated_code", ""),
        retry_count=final_state.get("retry_count", 0),
        error_trace=final_state.get("error_trace", "")
    )
