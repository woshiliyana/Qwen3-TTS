from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunQwen3LongformScriptTest(unittest.TestCase):
    def run_dry_run(
        self,
        env_overrides: dict[str, str],
        language: str = "Spanish",
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
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
                    language,
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

    def test_runner_root_is_derived_from_script_location(self) -> None:
        script_text = (ROOT / "tools" / "run_qwen3_longform.sh").read_text(encoding="utf-8")

        self.assertNotIn('ROOT="/Volumes/My Passport/vibe coding/Qwen3-TTS"', script_text)

    def test_spanish_uses_language_chunk_preset_by_default(self) -> None:
        result, _output_dir = self.run_dry_run({"LANG": "C", "LC_ALL": "C"}, language="Spanish")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("_420c_2048tok", result.stdout)
        self.assertIn("--max-chars 420", result.stdout)
        self.assertIn("--min-chars 220", result.stdout)

    def test_env_chunk_settings_override_language_preset(self) -> None:
        result, _output_dir = self.run_dry_run(
            {
                "LANG": "C",
                "LC_ALL": "C",
                "QWEN3_MAX_CHARS": "333",
                "QWEN3_MIN_CHARS": "111",
            },
            language="Spanish",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("_333c_2048tok", result.stdout)
        self.assertIn("--max-chars 333", result.stdout)
        self.assertIn("--min-chars 111", result.stdout)


class Qwen3CacheStatusScriptTest(unittest.TestCase):
    def test_lists_voice_cache_health_from_work_root(self) -> None:
        temp_root = ROOT / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(prefix="test-qwen3-cache-status-", dir=temp_root)
        self.addCleanup(temp_dir.cleanup)
        work_root = Path(temp_dir.name) / "qwen3_tts_work"
        complete_voice = work_root / "voices" / "spanish_abc123"
        incomplete_voice = work_root / "voices" / "japanese_def456"
        complete_voice.mkdir(parents=True)
        incomplete_voice.mkdir(parents=True)
        for filename in [
            "voice_clone_prompt.pt",
            "voice_clone_prompt.meta.json",
            "reference_24k_mono.wav",
            "reference_24k_mono.meta.json",
        ]:
            (complete_voice / filename).write_text("ok", encoding="utf-8")
        (incomplete_voice / "voice_clone_prompt.pt").write_text("ok", encoding="utf-8")

        env = os.environ.copy()
        env.update({"LANG": "C", "LC_ALL": "C", "QWEN3_WORK_ROOT": str(work_root)})

        result = subprocess.run(
            ["bash", str(ROOT / "tools" / "qwen3_cache_status.sh")],
            cwd=ROOT,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("spanish\tabc123\tok\tok\tok\tok\t", result.stdout)
        self.assertIn("japanese\tdef456\tok\tmissing\tmissing\tmissing\t", result.stdout)


if __name__ == "__main__":
    unittest.main()
