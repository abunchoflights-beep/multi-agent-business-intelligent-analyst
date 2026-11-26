import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.data_loader import (
    build_schema_context,
    ensure_tables_cached,
    parse_s3_location,
)


class TestDataLoader(unittest.TestCase):
    def test_parse_s3_location_defaults_and_override(self):
        bucket, prefix = parse_s3_location("")
        self.assertEqual(bucket, "david-bi-analyst-data-2026")
        self.assertEqual(prefix, "raw/owlist/ingest-date=2026-11-26")

        b2, p2 = parse_s3_location("s3://custom-bucket/custom/prefix/")
        self.assertEqual(b2, "custom-bucket")
        self.assertEqual(p2, "custom/prefix")

    def test_ensure_tables_cached_downloads_once_and_builds_schema(self):
        fake_s3 = MagicMock()

        def fake_download(bucket, key, dest_path):
            Path(dest_path).write_text(
                "product_category_name,product_category_name_english\n"
                "beleza_saude,health_beauty\n",
                encoding="utf-8",
            )

        fake_s3.download_file.side_effect = fake_download

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            ensure_tables_cached(
                fake_s3,
                "david-bi-analyst-data-2026",
                "raw/owlist/ingest-date=2026-11-26",
                ["category_translation", "nonexistent_table"],
                cache_dir=cache_dir,
            )
            self.assertEqual(fake_s3.download_file.call_count, 1)
            fake_s3.download_file.assert_called_with(
                "david-bi-analyst-data-2026",
                "raw/owlist/ingest-date=2026-11-26/product_category_name_translation.csv",
                str(cache_dir / ".product_category_name_translation.csv.tmp"),
            )

            # Second call should hit local disk cache instead of S3
            ensure_tables_cached(
                fake_s3,
                "david-bi-analyst-data-2026",
                "raw/owlist/ingest-date=2026-11-26",
                ["category_translation"],
                cache_dir=cache_dir,
            )
            self.assertEqual(fake_s3.download_file.call_count, 1)

            schema = build_schema_context(cache_dir, ["category_translation"])
            self.assertIn("datasets['category_translation']", schema)
            self.assertIn("product_category_name_english", schema)
            self.assertIn("beleza_saude", schema)


if __name__ == "__main__":
    unittest.main()
