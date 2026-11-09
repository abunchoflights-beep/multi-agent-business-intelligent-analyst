# src/app.py
import os
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI
from supabase import create_client, Client

# Load ONLY non-AWS 3rd-party secrets (like Supabase) from .env if present
load_dotenv()

app = FastAPI(
    title="Multi-Agent BI Analyst",
    description="Production-grade cloud-native Multi-Agent BI Analyst on AWS Bedrock & EC2"
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
    supabase = create_client(supabase_url, supabase_key)


@app.get("/health")
def health_check():
    """Liveness probe for Nginx, load balancers, and deployment verification."""
    return {
        "status": "healthy",
        "service": "bi-analyst-backend",
        "region": AWS_REGION,
        "supabase_configured": supabase is not None
    }
