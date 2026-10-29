import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uwseg import prepare_data


class DownloadTests(unittest.TestCase):
    def test_download_enforces_size_before_writing_extra_bytes(self):
        payload = b"123456789"
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(prepare_data.WORKSPACE))
            destination = Path(folder) / "sample.zip"
            with (
                patch.object(prepare_data, "ARCHIVE_BYTES", 8),
                patch.object(prepare_data.urllib.request, "urlopen", return_value=io.BytesIO(payload)),
            ):
                with self.assertRaisesRegex(ValueError, "exceeds pinned"):
                    prepare_data.obtain_archive(destination, True)
            self.assertFalse(destination.exists())
            self.assertLessEqual(destination.with_name("sample.zip.part").stat().st_size, 8)

    def test_download_checks_hash_then_renames(self):
        payload = b"12345678"
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(prepare_data.WORKSPACE))
            destination = Path(folder) / "sample.zip"
            with (
                patch.object(prepare_data, "ARCHIVE_BYTES", len(payload)),
                patch.object(prepare_data, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest()),
                patch.object(prepare_data.urllib.request, "urlopen", return_value=io.BytesIO(payload)),
            ):
                prepare_data.obtain_archive(destination, True)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_name("sample.zip.part").exists())


if __name__ == "__main__":
    unittest.main()
