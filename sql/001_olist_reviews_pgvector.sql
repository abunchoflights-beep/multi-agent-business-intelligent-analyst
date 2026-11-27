-- sql/001_olist_reviews_pgvector.sql
-- Run once in the BI Analyst Supabase project's SQL Editor.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.olist_review_embeddings (
    review_id text PRIMARY KEY,
    order_id text,
    review_score integer NOT NULL,
    product_category text,
    customer_state text,
    review_text text NOT NULL,
    content_hash text NOT NULL,
    embedding vector(512) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- HNSW cosine index works cleanly even when created before ingesting rows.
CREATE INDEX IF NOT EXISTS olist_reviews_embedding_hnsw_idx
    ON public.olist_review_embeddings
    USING hnsw (embedding vector_cosine_ops);

CREATE OR REPLACE FUNCTION public.match_olist_reviews(
    query_embedding vector(512),
    match_count integer DEFAULT 15,
    filter_min_score integer DEFAULT 1,
    filter_max_score integer DEFAULT 5
)
RETURNS TABLE (
    review_id text,
    order_id text,
    review_score integer,
    product_category text,
    customer_state text,
    review_text text,
    similarity double precision
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        r.review_id,
        r.order_id,
        r.review_score,
        r.product_category,
        r.customer_state,
        r.review_text,
        1 - (r.embedding <=> query_embedding) AS similarity
    FROM public.olist_review_embeddings AS r
    WHERE r.review_score BETWEEN filter_min_score AND filter_max_score
    ORDER BY r.embedding <=> query_embedding
    LIMIT match_count;
$$;
