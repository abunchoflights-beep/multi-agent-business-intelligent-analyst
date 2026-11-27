# src/agents/router_agent.py
import json
import re
from typing import List, Optional
from botocore.client import BaseClient
from supabase import Client

try:
    from src.data_loader import (
        OLIST_TABLES,
        build_schema_context,
        ensure_tables_cached,
        parse_s3_location,
    )
    from src.rag import format_reviews_context, retrieve_similar_reviews
    from src.state import AgentState
except ImportError:
    from data_loader import (
        OLIST_TABLES,
        build_schema_context,
        ensure_tables_cached,
        parse_s3_location,
    )
    from rag import format_reviews_context, retrieve_similar_reviews
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
            "min_score": 1,
            "max_score": 5,
            "reason": "Defaulted to compute routing with core Olist order tables.",
        }


def _select_valid_tables(raw_tables: object) -> List[str]:
    if not isinstance(raw_tables, list):
        return DEFAULT_TABLES
    valid = [t for t in raw_tables if isinstance(t, str) and t in OLIST_TABLES]
    # Cap at 4 tables to stay comfortably inside t3.micro memory limits.
    return valid[:4] if valid else DEFAULT_TABLES


def _clamp_score(val: object, default: int) -> int:
    try:
        n = int(val)  # type: ignore[arg-type]
        return max(1, min(5, n))
    except (TypeError, ValueError):
        return default


def intake_router_agent(
    state: AgentState,
    bedrock_client: BaseClient,
    s3_client: Optional[BaseClient] = None,
    supabase_client: Optional[Client] = None,
) -> dict:
    """
    Classifies intent ('compute', 'rag', or 'hybrid'):
    - For 'rag' or 'hybrid': queries Bedrock Titan V2 + Supabase pgvector for matching Olist reviews.
    - For 'compute' or 'hybrid' (or RAG fallback if pgvector is unseeded): caches selected S3 CSVs
      in the trusted parent process and injects schema + sample rows.
    """
    query = state.get("user_query", "")
    catalog_summary = "\n".join(
        f"- {name}: {meta['description']}" for name, meta in OLIST_TABLES.items()
    )

    prompt = f"""
    You are an Intake Router Agent for an Olist E-Commerce Business Intelligence system.
    Classify the query into one of three routes:
    - "compute": numerical aggregations, rankings, delivery rates, revenue, counts across CSV tables.
    - "rag": qualitative themes, complaint reasons, praise topics, or sentiment in customer reviews.
    - "hybrid": needs BOTH quantitative breakdown across CSVs AND qualitative review themes.

    Also choose:
    - "tables": 1 to 4 relevant table keys from the catalog below.
    - "min_score" / "max_score": review star filter (1-5). Use 1..2 if user asks about complaints/bad reviews, 4..5 for positive reviews, else 1..5.

    Available tables:
    {catalog_summary}

    Query: {query}

    Respond strictly in JSON with keys:
    - "route": "compute" | "rag" | "hybrid"
    - "tables": list of table keys from above
    - "min_score": integer 1-5
    - "max_score": integer 1-5
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
            "min_score": 1,
            "max_score": 5,
            "reason": "Fallback routing active.",
        }

    route = parsed.get("route", "compute")
    if route not in ("compute", "rag", "hybrid"):
        route = "compute"

    selected_tables = _select_valid_tables(parsed.get("tables"))
    min_score = _clamp_score(parsed.get("min_score"), 1)
    max_score = max(min_score, _clamp_score(parsed.get("max_score"), 5))

    rag_context = ""
    if route in ("rag", "hybrid"):
        try:
            matches = retrieve_similar_reviews(
                bedrock_client,
                supabase_client,
                query,
                match_count=15,
                min_score=min_score,
                max_score=max_score,
            )
            rag_context = format_reviews_context(matches)
        except Exception as exc:
            rag_context = ""
            raw_warning = f"RAG lookup fallback ({exc})"
        else:
            raw_warning = ""
    else:
        raw_warning = ""

    # If pure RAG found no pgvector rows (e.g., table not seeded yet), fall back to Pandas over order_reviews.
    if route == "rag" and not rag_context:
        route = "compute"
        if "order_reviews" not in selected_tables:
            selected_tables = ["order_reviews"] + selected_tables[:3]

    data_dir = ""
    raw_context = f"Route: {route} ({parsed.get('reason', '')})"
    if raw_warning:
        raw_context += f" [{raw_warning}]"

    if route in ("compute", "hybrid") and s3_client is not None:
        try:
            bucket, prefix = parse_s3_location(state.get("file_path", ""))
            cache_dir = ensure_tables_cached(s3_client, bucket, prefix, selected_tables)
            data_dir = str(cache_dir)
            schema_text = build_schema_context(cache_dir, selected_tables)
            raw_context = f"{raw_context}\n\n{schema_text}"
        except Exception as exc:
            raw_context = f"{raw_context}. S3 load warning: {exc}"

    return {
        "route": route,
        "selected_tables": selected_tables,
        "data_dir": data_dir,
        "raw_context": raw_context,
        "rag_context": rag_context,
        "retry_count": 0,
    }
