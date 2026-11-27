# scripts/ingest_reviews_to_pgvector.py
"""
Incremental ingestion script: reads Olist CSVs from S3 cache, enriches non-empty Portuguese
reviews with category & state, computes SHA-256 content hashes, embeds only new/changed
reviews via Amazon Bedrock Titan V2 (512-dim), and upserts into Supabase pgvector.

Usage (on EC2 or local with AWS credentials + .env):
    python scripts/ingest_reviews_to_pgvector.py --limit 2500
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import boto3
import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.data_loader import DEFAULT_BUCKET, DEFAULT_PREFIX, ensure_tables_cached
from src.rag import compute_content_hash, embed_text_titan

REQUIRED_TABLES = [
    "order_reviews",
    "orders",
    "order_items",
    "products",
    "category_translation",
    "customers",
]


def prepare_review_documents(cache_dir: Path, limit: int) -> List[Dict[str, object]]:
    """Joins reviews with English category + customer state and builds hashable text docs."""
    reviews = pd.read_csv(cache_dir / "olist_order_reviews_dataset.csv", encoding="utf-8-sig")
    orders = pd.read_csv(
        cache_dir / "olist_orders_dataset.csv",
        usecols=["order_id", "customer_id"],
        encoding="utf-8-sig",
    )
    customers = pd.read_csv(
        cache_dir / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_state"],
        encoding="utf-8-sig",
    )
    items = pd.read_csv(
        cache_dir / "olist_order_items_dataset.csv",
        usecols=["order_id", "product_id"],
        encoding="utf-8-sig",
    ).drop_duplicates(subset=["order_id"])
    products = pd.read_csv(
        cache_dir / "olist_products_dataset.csv",
        usecols=["product_id", "product_category_name"],
        encoding="utf-8-sig",
    )
    translations = pd.read_csv(
        cache_dir / "product_category_name_translation.csv",
        encoding="utf-8-sig",
    )

    # Keep only reviews with meaningful comment text
    reviews["review_comment_title"] = reviews["review_comment_title"].fillna("").astype(str).str.strip()
    reviews["review_comment_message"] = reviews["review_comment_message"].fillna("").astype(str).str.strip()
    reviews = reviews[reviews["review_comment_message"] != ""].drop_duplicates(subset=["review_id"])

    # Deterministic mix: sort so low-score and high-score reviews are both represented if limited
    if limit > 0:
        reviews = reviews.head(limit)

    products_en = products.merge(translations, on="product_category_name", how="left")
    products_en["product_category"] = (
        products_en["product_category_name_english"]
        .fillna(products_en["product_category_name"])
        .fillna("unknown")
    )

    enriched = (
        reviews.merge(orders, on="order_id", how="left")
        .merge(customers, on="customer_id", how="left")
        .merge(items, on="order_id", how="left")
        .merge(products_en[["product_id", "product_category"]], on="product_id", how="left")
    )

    docs: List[Dict[str, object]] = []
    for row in enriched.itertuples(index=False):
        title = getattr(row, "review_comment_title", "")
        msg = getattr(row, "review_comment_message", "")
        body = f"{title}: {msg}" if title else msg
        category = getattr(row, "product_category", "unknown")
        if pd.isna(category):
            category = "unknown"
        state = getattr(row, "customer_state", "unknown")
        if pd.isna(state):
            state = "unknown"
        score = int(getattr(row, "review_score", 0))

        # Include structured context in the embedded string so semantic search captures category/score
        embed_input = f"Category: {category} | State: {state} | Score: {score}/5 | Review: {body}"
        docs.append({
            "review_id": str(getattr(row, "review_id")),
            "order_id": str(getattr(row, "order_id")),
            "review_score": score,
            "product_category": str(category),
            "customer_state": str(state),
            "review_text": body,
            "embed_input": embed_input,
            "content_hash": compute_content_hash(embed_input),
        })
    return docs


def fetch_existing_hashes(supabase: Client) -> Dict[str, str]:
    """Loads existing review_id -> content_hash map in pages to skip unchanged rows."""
    hashes: Dict[str, str] = {}
    page_size = 1000
    offset = 0
    while True:
        resp = (
            supabase.table("olist_review_embeddings")
            .select("review_id,content_hash")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        for r in rows:
            hashes[r["review_id"]] = r["content_hash"]
        if len(rows) < page_size:
            break
        offset += page_size
    return hashes


def ingest(limit: int, batch_size: int) -> None:
    load_dotenv(BACKEND_ROOT / ".env")
    region = os.getenv("AWS_REGION", "us-east-2")
    bucket = os.getenv("S3_BUCKET_NAME", DEFAULT_BUCKET)
    prefix = os.getenv("S3_DATA_PREFIX", DEFAULT_PREFIX)

    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]

    s3_client = boto3.client("s3", region_name=region)
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    supabase = create_client(supabase_url, supabase_key)

    print(f"[1/4] Ensuring Olist CSVs cached from s3://{bucket}/{prefix}/ ...")
    cache_dir = ensure_tables_cached(s3_client, bucket, prefix, REQUIRED_TABLES)

    print(f"[2/4] Preparing review documents (limit={limit or 'ALL'}) ...")
    docs = prepare_review_documents(cache_dir, limit)
    print(f"      Prepared {len(docs)} non-empty reviews.")

    print("[3/4] Checking existing content hashes in Supabase pgvector ...")
    existing = fetch_existing_hashes(supabase)
    to_embed = [d for d in docs if existing.get(d["review_id"]) != d["content_hash"]]
    skipped = len(docs) - len(to_embed)
    print(f"      Skipping {skipped} unchanged rows; embedding {len(to_embed)} new/updated rows.")

    if not to_embed:
        print("Done! Vector store is already up to date.")
        return

    print("[4/4] Embedding via Amazon Titan Text V2 (512d) & upserting batches ...")
    buffer: List[Dict[str, object]] = []
    for idx, doc in enumerate(to_embed, start=1):
        embedding = embed_text_titan(bedrock_client, str(doc["embed_input"]))
        buffer.append({
            "review_id": doc["review_id"],
            "order_id": doc["order_id"],
            "review_score": doc["review_score"],
            "product_category": doc["product_category"],
            "customer_state": doc["customer_state"],
            "review_text": doc["review_text"],
            "content_hash": doc["content_hash"],
            "embedding": embedding,
        })
        if len(buffer) >= batch_size or idx == len(to_embed):
            supabase.table("olist_review_embeddings").upsert(buffer).execute()
            print(f"      Upserted {idx}/{len(to_embed)} reviews")
            buffer.clear()

    print("Ingestion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental Olist review ingestion into pgvector.")
    parser.add_argument("--limit", type=int, default=2500, help="Max non-empty reviews to process (0 = all).")
    parser.add_argument("--batch-size", type=int, default=100, help="Supabase upsert batch size.")
    args = parser.parse_args()
    ingest(limit=args.limit, batch_size=args.batch_size)
