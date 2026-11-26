import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.agents.executor import code_execution_node, _build_scrubbed_env


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

    def test_injects_preloaded_datasets_into_execute_analysis(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "product_category_name_translation.csv"
            csv_path.write_text(
                "product_category_name,product_category_name_english\n"
                "beleza_saude,health_beauty\n"
                "informatica_acessorios,computers_accessories\n",
                encoding="utf-8",
            )
            state = {
                "data_dir": tmp_dir,
                "selected_tables": ["category_translation"],
                "generated_code": """```python
def execute_analysis(datasets):
    rows = datasets["category_translation"]
    return {"row_count": len(rows)}
```""",
            }
            out = code_execution_node(state)
            self.assertEqual(out["error_trace"], "")
            self.assertIn('"row_count": 2', out["execution_result"])

    def test_blocks_pandas_read_csv_and_read_json(self):
        for expr in (
            'pd.read_csv("/home/ubuntu/bi-analyst-backend/.env")',
            'pd.read_json("http://169.254.169.254/latest/meta-data/")',
        ):
            state = {
                "generated_code": f"""```python
import pandas as pd

def execute_analysis(datasets):
    return {expr}
```"""
            }
            out = code_execution_node(state)
            self.assertIn("SandboxViolationError", out["error_trace"])

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
            env = _build_scrubbed_env()
            self.assertNotIn("SUPABASE_KEY", env)
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
        finally:
            os.environ.pop("SUPABASE_KEY", None)
            os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

    def test_blocks_infinite_loop_timeout(self):
        with patch("src.agents.executor.EXECUTION_TIMEOUT_SECONDS", 1):
            out = code_execution_node({
                "generated_code": "def execute_analysis():\n    while True: pass"
            })
        self.assertIn("TimeoutError", out["error_trace"])
        self.assertEqual(out["execution_result"], "")

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
