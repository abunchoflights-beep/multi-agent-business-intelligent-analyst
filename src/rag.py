# src/rag.py
import hashlib
import json
from typing import Any, Dict, List, Optional
from botocore.client import BaseClient
from supabase import Client

TITAN_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSIONS = 512


def compute_content_hash(text: str) -> str:
    """Stable SHA-256 digest used to skip re-embedding unchanged documents."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def embed_text_titan(
    bedrock_client: BaseClient,
    text: str,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> List[float]:
    """Generates a normalized vector embedding via Amazon Bedrock Titan Text Embeddings V2."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Cannot embed empty text.")

    # Guard against Titan V2 input token bounds on unusually long comments.
    truncated = cleaned[:4000]
    body = json.dumps({
        "inputText": truncated,
        "dimensions": dimensions,
        "normalize": True,
    })

    response = bedrock_client.invoke_model(
        modelId=TITAN_EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    payload = json.loads(response["body"].read().decode("utf-8"))
    return payload["embedding"]


def retrieve_similar_reviews(
    bedrock_client: BaseClient,
    supabase_client: Optional[Client],
    query: str,
    match_count: int = 15,
    min_score: int = 1,
    max_score: int = 5,
) -> List[Dict[str, Any]]:
    """Embeds the query with Titan V2 and fetches top-k matching Olist reviews from pgvector."""
    if supabase_client is None:
        return []

    query_embedding = embed_text_titan(bedrock_client, query)
    result = supabase_client.rpc(
        "match_olist_reviews",
        {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "filter_min_score": min_score,
            "filter_max_score": max_score,
        },
    ).execute()
    return result.data or []


def format_reviews_context(matches: List[Dict[str, Any]]) -> str:
    """Formats retrieved pgvector matches for downstream prompt injection."""
    if not matches:
        return ""

    lines = ["Top semantically matched customer reviews (via Bedrock Titan V2 + pgvector):"]
    for idx, row in enumerate(matches, start=1):
        score = row.get("review_score", "?")
        cat = row.get("product_category") or "unknown"
        state = row.get("customer_state") or "??"
        sim = row.get("similarity")
        sim_str = f"{sim:.3f}" if isinstance(sim, (int, float)) else "n/a"
        text = (row.get("review_text") or "").replace("\n", " ").strip()
        lines.append(f"{idx}. [score={score}/5 | category={cat} | state={state} | sim={sim_str}] {text}")
    return "\n".join(lines)
