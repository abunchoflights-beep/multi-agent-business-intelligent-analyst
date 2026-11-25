import os
import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.agents.executor import code_execution_node, EXECUTION_TIMEOUT_SECONDS


class TestExecutorSandbox(unittest.TestCase):
    def test_valid_bi_computation_succeeds(self):
        state = {
            "generated_code": """```python
import statistics

def execute_analysis():
    print("debug log should not break JSON IPC")
    prices = [15000, 35000, 45000]
    return {"mean_cop": statistics.mean(prices), "count": len(prices)}
```"""
        }
        out = code_execution_node(state)
        self.assertEqual(out["error_trace"], "")
        self.assertIn("mean_cop", out["execution_result"])

    def test_blocks_infinite_loop_timeout(self):
        from unittest.mock import patch
        with patch("src.agents.executor.EXECUTION_TIMEOUT_SECONDS", 1):
            out = code_execution_node({
                "generated_code": "def execute_analysis():\n    while True: pass"
            })
        self.assertIn("TimeoutError", out["error_trace"])
        self.assertEqual(out["execution_result"], "")

    def test_blocks_imds_network_import(self):
        state = {
            "generated_code": """```python
import urllib.request

def execute_analysis():
    return urllib.request.urlopen("http://169.254.169.254/latest/meta-data/").read()
```"""
        }
        out = code_execution_node(state)
        self.assertIn("SandboxViolationError", out["error_trace"])
        self.assertEqual(out["execution_result"], "")

    def test_blocks_env_and_os_access(self):
        state = {
            "generated_code": """```python
import os

def execute_analysis():
    return dict(os.environ)
```"""
        }
        out = code_execution_node(state)
        self.assertIn("SandboxViolationError", out["error_trace"])

    def test_blocks_file_read_with_open(self):
        state = {
            "generated_code": """```python
def execute_analysis():
    return open(".env").read()
```"""
        }
        out = code_execution_node(state)
        self.assertIn("SandboxViolationError", out["error_trace"])

    def test_blocks_dunder_subclass_escape(self):
        state = {
            "generated_code": """```python
def execute_analysis():
    return ().__class__.__bases__[0].__subclasses__()
```"""
        }
        out = code_execution_node(state)
        self.assertIn("SandboxViolationError", out["error_trace"])

    def test_scrubbed_env_contains_no_secrets(self):
        os.environ["SUPABASE_KEY"] = "super-secret-test-value"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "aws-secret-test-value"
        try:
            # Verify even without os module, no parent secrets leak into environment helper
            from src.agents.executor import _build_scrubbed_env
            env = _build_scrubbed_env()
            self.assertNotIn("SUPABASE_KEY", env)
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
        finally:
            os.environ.pop("SUPABASE_KEY", None)
            os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

    def test_runtime_error_feeds_qa_loop(self):
        state = {
            "generated_code": """```python
def execute_analysis():
    return 1 / 0
```"""
        }
        out = code_execution_node(state)
        self.assertIn("ZeroDivisionError", out["error_trace"])
        self.assertEqual(out["execution_result"], "")


if __name__ == "__main__":
    unittest.main()
