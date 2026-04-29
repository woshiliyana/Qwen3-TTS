from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunQwen3LongformScriptTest(unittest.TestCase):
    def run_dry_run(self, env_overrides: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], Path]:
        temp_root = ROOT / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(prefix="test-run-qwen3-longform-", dir=temp_root)
        self.addCleanup(temp_dir.cleanup)
        run_dir = Path(temp_dir.name)
        output_dir = run_dir / "episode-output"
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_audio = run_dir / "voice-reference.wav"
        text_path = output_dir / "episode-script.md"
        ref_audio.write_bytes(b"fake wav bytes")
        text_path.write_text("Uno. Dos. Tres.", encoding="utf-8")

        env = os.environ.copy()
        env.update(env_overrides)
        env["QWEN3_DRY_RUN"] = "1"

        result = subprocess.run(
            [
                "bash",
                str(ROOT / "tools" / "run_qwen3_longform.sh"),
                "Spanish",
                str(ref_audio),
                str(text_path),
            ],
            cwd=ROOT,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        return result, output_dir

    def test_default_work_dir_lives_under_project_root_not_output_dir(self) -> None:
        result, output_dir = self.run_dry_run({"LANG": "C", "LC_ALL": "C"})

        self.assertEqual(result.returncode, 0, result.stderr)
        expected_runs_prefix = str(ROOT / "qwen3_tts_work" / "runs")
        expected_voice_prefix = str(ROOT / "qwen3_tts_work" / "voices")
        self.assertIn(f"work_dir={expected_runs_prefix}/", result.stdout)
        self.assertIn(f"voice_cache_dir={expected_voice_prefix}/spanish_", result.stdout)
        self.assertNotIn(f"work_dir={output_dir}/", result.stdout)
        self.assertIn("--voice-cache-dir", result.stdout)

    def test_dry_run_hashing_is_not_broken_by_invalid_utf8_locale(self) -> None:
        result, _output_dir = self.run_dry_run({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("work_dir=", result.stdout)


if __name__ == "__main__":
    unittest.main()
