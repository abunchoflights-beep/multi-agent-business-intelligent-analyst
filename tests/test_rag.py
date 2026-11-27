import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.agents.report_agent import report_synthesis_agent
from src.agents.router_agent import intake_router_agent
from src.rag import (
    EMBEDDING_DIMENSIONS,
    TITAN_EMBED_MODEL_ID,
    compute_content_hash,
    embed_text_titan,
    format_reviews_context,
    retrieve_similar_reviews,
)


def _mock_bedrock_response(payload: dict) -> dict:
    return {"body": io.BytesIO(json.dumps(payload).encode("utf-8"))}


class TestBedrockTitanRag(unittest.TestCase):
    def test_content_hash_is_deterministic_and_sensitive(self):
        h1 = compute_content_hash("Produto chegou atrasado")
        h2 = compute_content_hash("Produto chegou atrasado")
        h3 = compute_content_hash("Produto chegou no prazo")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertEqual(len(h1), 32)

    def test_embed_text_titan_payload_and_response(self):
        fake_bedrock = MagicMock()
        fake_vector = [0.01] * EMBEDDING_DIMENSIONS
        fake_bedrock.invoke_model.return_value = _mock_bedrock_response({"embedding": fake_vector})

        vec = embed_text_titan(fake_bedrock, "  Entrega muito atrasada  ")
        self.assertEqual(len(vec), EMBEDDING_DIMENSIONS)

        call_kwargs = fake_bedrock.invoke_model.call_args.kwargs
        self.assertEqual(call_kwargs["modelId"], TITAN_EMBED_MODEL_ID)
        sent_body = json.loads(call_kwargs["body"])
        self.assertEqual(sent_body["inputText"], "Entrega muito atrasada")
        self.assertEqual(sent_body["dimensions"], EMBEDDING_DIMENSIONS)
        self.assertTrue(sent_body["normalize"])

    def test_retrieve_similar_reviews_calls_pgvector_rpc(self):
        fake_bedrock = MagicMock()
        fake_bedrock.invoke_model.return_value = _mock_bedrock_response(
            {"embedding": [0.1] * EMBEDDING_DIMENSIONS}
        )
        fake_supabase = MagicMock()
        fake_supabase.rpc.return_value.execute.return_value.data = [
            {
                "review_id": "r1",
                "review_score": 1,
                "product_category": "furniture_decor",
                "customer_state": "RJ",
                "review_text": "Produto veio quebrado e atrasado.",
                "similarity": 0.872,
            }
        ]

        matches = retrieve_similar_reviews(
            fake_bedrock, fake_supabase, "broken items", match_count=10, min_score=1, max_score=2
        )
        self.assertEqual(len(matches), 1)
        fake_supabase.rpc.assert_called_once()
        rpc_name, rpc_params = fake_supabase.rpc.call_args.args
        self.assertEqual(rpc_name, "match_olist_reviews")
        self.assertEqual(rpc_params["filter_min_score"], 1)
        self.assertEqual(rpc_params["filter_max_score"], 2)

        formatted = format_reviews_context(matches)
        self.assertIn("furniture_decor", formatted)
        self.assertIn("sim=0.872", formatted)

    def test_router_selects_rag_when_pgvector_returns_matches(self):
        fake_bedrock = MagicMock()
        router_json = json.dumps({
            "route": "rag",
            "tables": ["order_reviews"],
            "min_score": 1,
            "max_score": 2,
            "reason": "Semantic complaint themes in negative reviews.",
        })
        fake_bedrock.invoke_model.side_effect = [
            _mock_bedrock_response({"output": {"message": {"content": [{"text": router_json}]}}}),
            _mock_bedrock_response({"embedding": [0.2] * EMBEDDING_DIMENSIONS}),
        ]
        fake_supabase = MagicMock()
        fake_supabase.rpc.return_value.execute.return_value.data = [
            {
                "review_id": "r2",
                "review_score": 1,
                "product_category": "telephony",
                "customer_state": "SP",
                "review_text": "Nao recebi o carregador.",
                "similarity": 0.81,
            }
        ]

        out = intake_router_agent(
            {"user_query": "Why are 1-2 star customers complaining?", "file_path": ""},
            bedrock_client=fake_bedrock,
            s3_client=None,
            supabase_client=fake_supabase,
        )
        self.assertEqual(out["route"], "rag")
        self.assertIn("Nao recebi o carregador", out["rag_context"])

    def test_router_falls_back_to_compute_when_pgvector_unseeded(self):
        fake_bedrock = MagicMock()
        router_json = json.dumps({
            "route": "rag",
            "tables": ["orders"],
            "min_score": 1,
            "max_score": 2,
            "reason": "Review themes",
        })
        fake_bedrock.invoke_model.return_value = _mock_bedrock_response(
            {"output": {"message": {"content": [{"text": router_json}]}}}
        )

        out = intake_router_agent(
            {"user_query": "Why are 1-star reviews unhappy?", "file_path": ""},
            bedrock_client=fake_bedrock,
            s3_client=None,
            supabase_client=None,
        )
        self.assertEqual(out["route"], "compute")
        self.assertEqual(out["rag_context"], "")
        self.assertIn("order_reviews", out["selected_tables"])

    def test_report_synthesizes_pure_rag_route(self):
        fake_bedrock = MagicMock()
        fake_bedrock.invoke_model.return_value = _mock_bedrock_response(
            {"output": {"message": {"content": [{"text": "Main theme: missing items and shipping delays."}]}}}
        )
        out = report_synthesis_agent(
            {
                "user_query": "Top complaints?",
                "route": "rag",
                "selected_tables": ["order_reviews"],
                "execution_result": "",
                "rag_context": "1. [score=1/5 | category=telephony] Nao recebi.",
                "error_trace": "",
                "retry_count": 0,
            },
            fake_bedrock,
        )
        self.assertIn("missing items", out["final_report"])


if __name__ == "__main__":
    unittest.main()
