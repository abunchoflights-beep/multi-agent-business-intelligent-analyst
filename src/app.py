# src/app.py
import os
import sys
from pathlib import Path
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from supabase import create_client, Client

# Ensure project root is in sys.path so module imports resolve cleanly across environments
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.executor import get_sandbox_mode
from src.data_loader import DEFAULT_BUCKET, DEFAULT_PREFIX
from src.graph import create_bi_analyst_graph

# Load ONLY non-AWS 3rd-party secrets (like Supabase) from .env if present
load_dotenv()

app = FastAPI(
    title="Multi-Agent BI Analyst",
    description="Production-grade cloud-native Multi-Agent BI Analyst on AWS Bedrock & EC2",
    version="1.0.0",
)

# Notice: NO aws_access_key_id or aws_secret_access_key passed here!
# Boto3 resolves credentials via ~/.aws/credentials locally or EC2 IAM Role via IMDSv2.
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
)

s3_client = boto3.client(
    service_name="s3",
    region_name=AWS_REGION,
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
bi_analyst_graph = create_bi_analyst_graph(bedrock_runtime, s3_client, supabase)


class AnalysisRequest(BaseModel):
    user_query: str = Field(
        ...,
        description="The business question or analytical request",
        example="Which 5 product categories have the worst late-delivery rate?",
    )
    file_path: str = Field(
        default="",
        description="Optional S3 URI override (defaults to configured Olist snapshot)",
        example=f"s3://{DEFAULT_BUCKET}/{DEFAULT_PREFIX}/",
    )
    session_id: str = Field(
        default="session-001",
        description="Session identifier for multi-turn state tracking",
    )


class AnalysisResponse(BaseModel):
    session_id: str
    user_query: str
    route: str = "compute"
    selected_tables: list[str] = []
    execution_result: str
    rag_context: str = ""
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
        "s3_dataset": f"s3://{DEFAULT_BUCKET}/{DEFAULT_PREFIX}/",
        "sandbox_isolation": get_sandbox_mode(),
        "supabase_configured": supabase is not None,
    }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_data(request: AnalysisRequest):
    """
    Executes the multi-agent BI workflow:
    1. Router Agent: Classifies intent ('compute', 'rag', 'hybrid'), queries pgvector and/or caches S3 CSVs.
    2. Analyst Agent: Generates Python transformation code over pre-loaded `datasets`.
    3. Sandbox Executor: Runs code in an isolated, credential-free subprocess.
    4. QA Hotfix Loop: Catches any runtime exceptions and heals the code.
    5. Report Agent: Synthesizes quantitative + semantic evidence into an executive summary.
    """
    initial_state = {
        "session_id": request.session_id,
        "user_query": request.user_query,
        "file_path": request.file_path,
        "route": "",
        "selected_tables": [],
        "data_dir": "",
        "raw_context": "",
        "rag_context": "",
        "generated_code": "",
        "execution_result": "",
        "final_report": "",
        "error_trace": "",
        "retry_count": 0,
    }

    final_state = bi_analyst_graph.invoke(initial_state)

    return AnalysisResponse(
        session_id=final_state.get("session_id", request.session_id),
        user_query=final_state.get("user_query", request.user_query),
        route=final_state.get("route", "compute"),
        selected_tables=final_state.get("selected_tables", []),
        execution_result=final_state.get("execution_result", ""),
        rag_context=final_state.get("rag_context", ""),
        final_report=final_state.get("final_report", ""),
        generated_code=final_state.get("generated_code", ""),
        retry_count=final_state.get("retry_count", 0),
        error_trace=final_state.get("error_trace", ""),
    )
