# src/agents/router_agent.py
import json
import re
from typing import List, Optional
from botocore.client import BaseClient

try:
    from src.data_loader import (
        OLIST_TABLES,
        build_schema_context,
        ensure_tables_cached,
        parse_s3_location,
    )
    from src.state import AgentState
except ImportError:
    from data_loader import (
        OLIST_TABLES,
        build_schema_context,
        ensure_tables_cached,
        parse_s3_location,
    )
    from state import AgentState

DEFAULT_TABLES = ["orders", "order_items", "products", "category_translation"]


def clean_json_response(raw_text: str) -> dict:
    """Strips markdown code fences and parses JSON safely."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    cleaned = match.group(1).strip() if match else raw_text.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {
            "route": "compute",
            "tables": DEFAULT_TABLES,
            "reason": "Defaulted to compute routing with core Olist order tables.",
        }


def _select_valid_tables(raw_tables: object) -> List[str]:
    if not isinstance(raw_tables, list):
        return DEFAULT_TABLES
    valid = [t for t in raw_tables if isinstance(t, str) and t in OLIST_TABLES]
    # Cap at 4 tables to stay comfortably inside t3.micro memory limits.
    return valid[:4] if valid else DEFAULT_TABLES


def intake_router_agent(
    state: AgentState,
    bedrock_client: BaseClient,
    s3_client: Optional[BaseClient] = None,
) -> dict:
    """
    Classifies user intent, selects relevant Olist tables, fetches missing CSVs from S3
    in the trusted parent process, and injects real CSV column headers + sample rows.
    """
    query = state.get("user_query", "")
    catalog_summary = "\n".join(
        f"- {name}: {meta['description']}" for name, meta in OLIST_TABLES.items()
    )

    prompt = f"""
    You are an Intake Router Agent for an Olist E-Commerce Business Intelligence system.
    Determine whether the user query requires 'compute' (Pandas metrics/aggregations),
    'rag' (Portuguese customer review sentiment/themes), or 'hybrid' (both), and pick
    the 1 to 4 most relevant tables from the catalog below.

    Available tables:
    {catalog_summary}

    Query: {query}

    Respond strictly in JSON with keys:
    - "route": "compute" | "rag" | "hybrid"
    - "tables": list of table keys from above
    - "reason": short explanation
    """

    try:
        response = bedrock_client.invoke_model(
            modelId="amazon.nova-micro-v1:0",
            body=json.dumps({
                "inferenceConfig": {"temperature": 0.0},
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
            }),
        )
        result = json.loads(response["body"].read().decode("utf-8"))
        raw_text = result["output"]["message"]["content"][0]["text"]
        parsed = clean_json_response(raw_text)
    except Exception:
        parsed = {
            "route": "compute",
            "tables": DEFAULT_TABLES,
            "reason": "Fallback routing active.",
        }

    route = parsed.get("route", "compute")
    selected_tables = _select_valid_tables(parsed.get("tables"))
    data_dir = ""
    raw_context = f"Route: {route} ({parsed.get('reason', '')})"

    if s3_client is not None:
        try:
            bucket, prefix = parse_s3_location(state.get("file_path", ""))
            cache_dir = ensure_tables_cached(s3_client, bucket, prefix, selected_tables)
            data_dir = str(cache_dir)
            schema_text = build_schema_context(cache_dir, selected_tables)
            raw_context = f"Route: {route} ({parsed.get('reason', '')})\n\n{schema_text}"
        except Exception as exc:
            raw_context = f"Route: {route}. S3 load warning: {exc}"

    return {
        "route": route,
        "selected_tables": selected_tables,
        "data_dir": data_dir,
        "raw_context": raw_context,
        "retry_count": 0,
    }
