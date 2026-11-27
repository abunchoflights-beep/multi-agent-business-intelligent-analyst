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


class TestReportAgent(unittest.TestCase):
    def test_synthesizes_executive_summary_from_bedrock(self):
        fake_bedrock = MagicMock()
        body_bytes = json.dumps({
            "output": {
                "message": {
                    "content": [{"text": "Direct answer: furniture had the highest delay rate."}]
                }
            }
        }).encode("utf-8")
        fake_bedrock.invoke_model.return_value = {"body": io.BytesIO(body_bytes)}

        out = report_synthesis_agent(
            {
                "user_query": "Which category is slowest?",
                "route": "compute",
                "selected_tables": ["orders"],
                "execution_result": '[{"category": "furniture", "rate": 12.5}]',
                "error_trace": "",
                "retry_count": 0,
            },
            fake_bedrock,
        )
        self.assertIn("furniture had the highest delay rate", out["final_report"])

    def test_surfaces_failure_summary_when_retries_exhausted(self):
        fake_bedrock = MagicMock()
        out = report_synthesis_agent(
            {
                "user_query": "Broken query",
                "execution_result": "",
                "error_trace": "KeyError: 'missing_col'",
                "retry_count": 3,
            },
            fake_bedrock,
        )
        fake_bedrock.invoke_model.assert_not_called()
        self.assertIn("failed after 3 QA hotfix attempts", out["final_report"])
        self.assertIn("KeyError", out["final_report"])


if __name__ == "__main__":
    unittest.main()
