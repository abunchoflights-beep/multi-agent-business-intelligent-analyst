# src/data_loader.py
import csv
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from botocore.client import BaseClient

DEFAULT_BUCKET = os.getenv("S3_BUCKET_NAME", "david-bi-analyst-data-2026")
DEFAULT_PREFIX = os.getenv("S3_DATA_PREFIX", "raw/owlist/ingest-date=2026-11-26")

# Registry of curated Olist tables stored in S3.
# Keeping an explicit allowlist prevents arbitrary S3 key or path traversal.
OLIST_TABLES: Dict[str, Dict[str, str]] = {
    "orders": {
        "filename": "olist_orders_dataset.csv",
        "description": "Order lifecycle timestamps, status, and customer_id (~99.4k rows).",
    },
    "order_items": {
        "filename": "olist_order_items_dataset.csv",
        "description": "Line items per order with product_id, seller_id, price, and freight_value (~112.6k rows).",
    },
    "order_payments": {
        "filename": "olist_order_payments_dataset.csv",
        "description": "Payment types, installments, and payment_value per order (~103.8k rows).",
    },
    "order_reviews": {
        "filename": "olist_order_reviews_dataset.csv",
        "description": "1-5 review scores and Portuguese customer review comments (~99k+ reviews).",
    },
    "products": {
        "filename": "olist_products_dataset.csv",
        "description": "Product categories and physical dimensions (~32.9k rows).",
    },
    "customers": {
        "filename": "olist_customers_dataset.csv",
        "description": "Customer IDs, zip prefixes, cities, and Brazilian states (~99.4k rows).",
    },
    "sellers": {
        "filename": "olist_sellers_dataset.csv",
        "description": "Seller IDs, zip prefixes, cities, and states (~3.1k rows).",
    },
    "category_translation": {
        "filename": "product_category_name_translation.csv",
        "description": "Maps Portuguese product_category_name to English product_category_name_english (71 rows).",
    },
}


def get_default_cache_dir() -> Path:
    """
    Returns a staging directory outside /home/ubuntu so an unprivileged sandbox user
    can read cached public CSVs (0755/0644) without traversing /home/ubuntu or .env.
    """
    configured = os.getenv("DATA_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "bi_analyst_olist_cache"


def parse_s3_location(file_path: str) -> Tuple[str, str]:
    """Resolves an optional s3://bucket/prefix URI, falling back to configured defaults."""
    cleaned = (file_path or "").strip()
    if cleaned.startswith("s3://"):
        without_scheme = cleaned[5:]
        parts = without_scheme.split("/", 1)
        bucket = parts[0] or DEFAULT_BUCKET
        prefix = parts[1].strip("/") if len(parts) > 1 else DEFAULT_PREFIX
        return bucket, prefix
    return DEFAULT_BUCKET, DEFAULT_PREFIX.strip("/")


def ensure_tables_cached(
    s3_client: BaseClient,
    bucket: str,
    prefix: str,
    tables: List[str],
    cache_dir: Path | None = None,
) -> Path:
    """
    Privilege-separated S3 fetch: runs ONLY in the trusted parent process (which holds
    the EC2 IAM role via IMDSv2) and stores public CSVs in a read-only staging cache.
    """
    target_dir = cache_dir or get_default_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(target_dir, 0o755)

    for table in tables:
        meta = OLIST_TABLES.get(table)
        if not meta:
            continue
        filename = meta["filename"]
        local_path = target_dir / filename

        if not local_path.exists() or local_path.stat().st_size == 0:
            s3_key = f"{prefix}/{filename}" if prefix else filename
            tmp_path = target_dir / f".{filename}.tmp"
            s3_client.download_file(bucket, s3_key, str(tmp_path))
            os.replace(tmp_path, local_path)

        if os.name == "posix":
            os.chmod(local_path, 0o644)

    return target_dir


def build_schema_context(cache_dir: Path, tables: List[str], sample_rows: int = 2) -> str:
    """
    Reads CSV headers and a tiny row sample from disk without loading full DataFrames
    into FastAPI's memory, giving the Analyst Agent exact column spellings.
    """
    sections: List[str] = []
    for table in tables:
        meta = OLIST_TABLES.get(table)
        if not meta:
            continue
        csv_path = cache_dir / meta["filename"]
        if not csv_path.exists():
            continue

        with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            samples = []
            for _ in range(sample_rows):
                row = next(reader, None)
                if row is None:
                    break
                samples.append(dict(zip(headers, row)))

        sections.append(
            f"Table `datasets['{table}']` ({meta['filename']}):\n"
            f"  Description: {meta['description']}\n"
            f"  Columns: {headers}\n"
            f"  Sample rows: {samples}"
        )

    return "\n\n".join(sections)
