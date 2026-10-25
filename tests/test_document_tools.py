"""Check preservation of edited reports and independently verified archive payloads."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


TOOLS = Path(__file__).resolve().parents[1] / "tools"


class DocumentToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tools").mkdir()
        for name in ["run_generated.py", "package_submission.py"]:
            shutil.copyfile(TOOLS / name, self.root / "tools" / name)
        (self.root / "README.md").write_text("Edited report\n", encoding="utf-8")

    def run_tool(self, name, *args):
        return subprocess.run([sys.executable, str(self.root / "tools" / name), *args], capture_output=True, text=True)

    def test_generated_report_keeps_edits_and_rebases_links(self):
        result = self.run_tool("run_generated.py", "--output", "README.md", "--", sys.executable, "-c",
            "from pathlib import Path; Path('README.md').write_text('[Metrics](artifacts/results.json)')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "README.md").read_text(), "Edited report\n")
        self.assertEqual((self.root / "output/generated/README.md").read_text(), "[Metrics](../../artifacts/results.json)")

    def test_failed_generator_restores_original_report(self):
        result = self.run_tool("run_generated.py", "--output", "README.md", "--", sys.executable, "-c",
            "from pathlib import Path; Path('README.md').write_text('incomplete'); raise SystemExit(7)")
        self.assertEqual(result.returncode, 7)
        self.assertEqual((self.root / "README.md").read_text(), "Edited report\n")
        self.assertFalse((self.root / "output/generated/README.md").exists())

    def test_package_excludes_caches_and_detects_changed_payload(self):
        for name in ["artifacts/main/cache", "artifacts/main/models", "report/build"]:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts/main/cache/large.npz").write_bytes(b"cache")
        (self.root / "artifacts/main/models/model.pt").write_bytes(b"weights")
        (self.root / "report/report.pdf").write_bytes(b"previous export")
        (self.root / "report/main.tex").write_text("Current source\n")
        result = self.run_tool("package_submission.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        path = self.root / "submission/project.zip"
        with zipfile.ZipFile(path) as archive:
            payload = {n: archive.read(n) for n in archive.namelist()}
        self.assertTrue(any(n.endswith("/report/main.tex") for n in payload))
        self.assertFalse(any("/cache/" in n or "/models/" in n or n.endswith(".pdf") for n in payload))
        readme = next(n for n in payload if n.endswith("/README.md"))
        payload[readme] = b"changed"
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in payload.items():
                archive.writestr(name, content)
        result = self.run_tool("package_submission.py", "--verify", str(path))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Checksum mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
