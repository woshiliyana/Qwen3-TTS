from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunQwen3LongformScriptTest(unittest.TestCase):
    def make_fixture(self) -> dict[str, Path]:
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
        return {
            "run_dir": run_dir,
            "output_dir": output_dir,
            "ref_audio": ref_audio,
            "text_path": text_path,
            "work_root": run_dir / "qwen3_tts_work",
        }

    def run_wrapper(
        self,
        fixture: dict[str, Path],
        env_overrides: dict[str, str],
        language: str = "Spanish",
        dry_run: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(env_overrides)
        if dry_run:
            env["QWEN3_DRY_RUN"] = "1"

        return subprocess.run(
            [
                "bash",
                str(ROOT / "tools" / "run_qwen3_longform.sh"),
                language,
                str(fixture["ref_audio"]),
                str(fixture["text_path"]),
            ],
            cwd=ROOT,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )

    def read_registry(self, work_root: Path) -> list[dict[str, object]]:
        registry_path = work_root / "runs" / "index.jsonl"
        if not registry_path.exists():
            return []
        return [
            json.loads(line)
            for line in registry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def write_fake_runner(self, run_dir: Path, exit_code: int = 0) -> Path:
        runner_path = run_dir / f"fake_runner_{exit_code}.py"
        runner_path.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import argparse
                import json
                import sys
                from pathlib import Path

                parser = argparse.ArgumentParser()
                parser.add_argument("--work-dir", required=True)
                parser.add_argument("--output-dir", required=True)
                parser.add_argument("--output-name", required=True)
                args, _unknown = parser.parse_known_args()

                work_dir = Path(args.work_dir)
                output_dir = Path(args.output_dir)
                work_dir.mkdir(parents=True, exist_ok=True)
                output_dir.mkdir(parents=True, exist_ok=True)
                final_output = output_dir / args.output_name
                segment_health = work_dir / "segment_health_report.json"
                quality_report = work_dir / "quality_report.json"
                segment_health.write_text("{{}}", encoding="utf-8")
                quality_report.write_text("{{}}", encoding="utf-8")
                final_output.write_bytes(b"fake wav")
                (work_dir / "manifest.json").write_text(
                    json.dumps(
                        {{
                            "final_output": str(final_output),
                            "segment_health_report": str(segment_health),
                            "quality_report": str(quality_report),
                        }}
                    ),
                    encoding="utf-8",
                )
                sys.exit({exit_code})
                """
            ),
            encoding="utf-8",
        )
        runner_path.chmod(0o755)
        return runner_path

    def run_dry_run(
        self,
        env_overrides: dict[str, str],
        language: str = "Spanish",
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        fixture = self.make_fixture()
        env = {"QWEN3_RECORD_DRY_RUN": "0", **env_overrides}
        result = self.run_wrapper(fixture, env, language=language, dry_run=True)
        output_dir = fixture["output_dir"]
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

    def test_dry_run_appends_registry_record_by_default(self) -> None:
        fixture = self.make_fixture()
        work_root = fixture["work_root"]

        result = self.run_wrapper(
            fixture,
            {"LANG": "C", "LC_ALL": "C", "QWEN3_WORK_ROOT": str(work_root)},
            dry_run=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.read_registry(work_root)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["event"], "planned")
        self.assertEqual(record["mode"], "dry_run")
        self.assertEqual(record["status"], "planned")
        self.assertEqual(record["language"], "Spanish")
        self.assertEqual(record["text_path"], str(fixture["text_path"]))
        self.assertEqual(record["ref_audio"], str(fixture["ref_audio"]))
        self.assertEqual(record["duration_s"], None)
        self.assertEqual(record["exit_code"], None)
        self.assertIn("tools/qwen3_longform_es.py", record["command_preview"])
        self.assertIn("git_sha", record)
        self.assertFalse((work_root / "run.lock.d").exists())

    def test_dry_run_registry_record_can_be_skipped(self) -> None:
        fixture = self.make_fixture()
        work_root = fixture["work_root"]

        result = self.run_wrapper(
            fixture,
            {
                "LANG": "C",
                "LC_ALL": "C",
                "QWEN3_WORK_ROOT": str(work_root),
                "QWEN3_RECORD_DRY_RUN": "0",
            },
            dry_run=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.read_registry(work_root), [])
        self.assertFalse((work_root / "run.lock.d").exists())

    def test_fake_real_run_appends_started_and_finished_records(self) -> None:
        fixture = self.make_fixture()
        work_root = fixture["work_root"]
        fake_runner = self.write_fake_runner(fixture["run_dir"], exit_code=0)

        result = self.run_wrapper(
            fixture,
            {
                "LANG": "C",
                "LC_ALL": "C",
                "QWEN3_WORK_ROOT": str(work_root),
                "QWEN3_SKIP_ENV_CHECK": "1",
                "QWEN3_RUNNER_BIN": str(fake_runner),
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((work_root / "run.lock.d").exists())
        records = self.read_registry(work_root)
        self.assertEqual([record["event"] for record in records], ["started", "finished"])
        self.assertEqual(records[0]["status"], "running")
        self.assertEqual(records[1]["status"], "completed")
        self.assertEqual(records[1]["exit_code"], 0)
        self.assertIsInstance(records[1]["duration_s"], float)
        self.assertTrue(str(records[1]["manifest_path"]).endswith("/manifest.json"))
        self.assertTrue(str(records[1]["final_output"]).endswith("_qwen3_spanish_smooth.wav"))
        self.assertTrue(str(records[1]["segment_health_report"]).endswith("/segment_health_report.json"))
        self.assertTrue(str(records[1]["quality_report"]).endswith("/quality_report.json"))

    def test_fake_real_run_failure_records_failed_and_releases_lock(self) -> None:
        fixture = self.make_fixture()
        work_root = fixture["work_root"]
        fake_runner = self.write_fake_runner(fixture["run_dir"], exit_code=7)

        result = self.run_wrapper(
            fixture,
            {
                "LANG": "C",
                "LC_ALL": "C",
                "QWEN3_WORK_ROOT": str(work_root),
                "QWEN3_SKIP_ENV_CHECK": "1",
                "QWEN3_RUNNER_BIN": str(fake_runner),
            },
        )

        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertFalse((work_root / "run.lock.d").exists())
        records = self.read_registry(work_root)
        self.assertEqual([record["event"] for record in records], ["started", "finished"])
        self.assertEqual(records[1]["status"], "failed")
        self.assertEqual(records[1]["exit_code"], 7)

    def test_active_lock_blocks_real_run_and_records_lock_holder(self) -> None:
        fixture = self.make_fixture()
        work_root = fixture["work_root"]
        lock_dir = work_root / "run.lock.d"
        lock_dir.mkdir(parents=True)
        holder = {
            "run_id": "existing-run",
            "pid": os.getpid(),
            "created_at": "2026-04-30T00:00:00-0700",
            "language": "Spanish",
            "text_path": "/Volumes/example/script.md",
            "work_dir": "/Volumes/example/work",
            "voice_cache_dir": "/Volumes/example/voice",
        }
        (lock_dir / "lock.json").write_text(json.dumps(holder), encoding="utf-8")
        fake_runner = self.write_fake_runner(fixture["run_dir"], exit_code=0)

        result = self.run_wrapper(
            fixture,
            {
                "LANG": "C",
                "LC_ALL": "C",
                "QWEN3_WORK_ROOT": str(work_root),
                "QWEN3_SKIP_ENV_CHECK": "1",
                "QWEN3_RUNNER_BIN": str(fake_runner),
            },
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(lock_dir.exists())
        self.assertEqual(json.loads((lock_dir / "lock.json").read_text(encoding="utf-8")), holder)
        records = self.read_registry(work_root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event"], "blocked")
        self.assertEqual(records[0]["status"], "blocked")
        self.assertEqual(records[0]["lock_holder"], holder)

    def test_stale_lock_is_archived_then_replaced_for_real_run(self) -> None:
        fixture = self.make_fixture()
        work_root = fixture["work_root"]
        lock_dir = work_root / "run.lock.d"
        lock_dir.mkdir(parents=True)
        (lock_dir / "lock.json").write_text(
            json.dumps({"run_id": "stale-run", "pid": 999999999}),
            encoding="utf-8",
        )
        fake_runner = self.write_fake_runner(fixture["run_dir"], exit_code=0)

        result = self.run_wrapper(
            fixture,
            {
                "LANG": "C",
                "LC_ALL": "C",
                "QWEN3_WORK_ROOT": str(work_root),
                "QWEN3_SKIP_ENV_CHECK": "1",
                "QWEN3_RUNNER_BIN": str(fake_runner),
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(lock_dir.exists())
        archived_locks = list((work_root / "stale-locks").glob("run.lock.d.*"))
        self.assertEqual(len(archived_locks), 1)
        self.assertTrue((archived_locks[0] / "lock.json").exists())
        self.assertEqual([record["event"] for record in self.read_registry(work_root)], ["started", "finished"])

    def test_malformed_lock_is_archived_then_replaced_for_real_run(self) -> None:
        fixture = self.make_fixture()
        work_root = fixture["work_root"]
        lock_dir = work_root / "run.lock.d"
        lock_dir.mkdir(parents=True)
        (lock_dir / "lock.json").write_text("{not-json", encoding="utf-8")
        fake_runner = self.write_fake_runner(fixture["run_dir"], exit_code=0)

        result = self.run_wrapper(
            fixture,
            {
                "LANG": "C",
                "LC_ALL": "C",
                "QWEN3_WORK_ROOT": str(work_root),
                "QWEN3_SKIP_ENV_CHECK": "1",
                "QWEN3_RUNNER_BIN": str(fake_runner),
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(lock_dir.exists())
        archived_locks = list((work_root / "stale-locks").glob("run.lock.d.*"))
        self.assertEqual(len(archived_locks), 1)
        self.assertEqual((archived_locks[0] / "lock.json").read_text(encoding="utf-8"), "{not-json")


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


class Qwen3EnvCheckScriptTest(unittest.TestCase):
    def test_env_check_succeeds_against_fake_project_root_without_model_load(self) -> None:
        temp_root = ROOT / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(prefix="test-qwen3-env-check-", dir=temp_root)
        self.addCleanup(temp_dir.cleanup)
        fake_root = Path(temp_dir.name) / "fake-root"
        python_bin = fake_root / ".venv-qwen-prod" / "bin" / "python"
        model_dir = fake_root / "models" / "Qwen3-TTS-12Hz-1.7B-Base-local"
        fake_bin = fake_root / "fake-bin"
        python_bin.parent.mkdir(parents=True)
        model_dir.mkdir(parents=True)
        fake_bin.mkdir(parents=True)
        os.symlink(sys.executable, python_bin)
        for tool_name in ("ffmpeg", "ffprobe"):
            tool_path = fake_bin / tool_name
            tool_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            tool_path.chmod(0o755)

        env = os.environ.copy()
        env.update(
            {
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
                "QWEN3_ENV_CHECK_TEST_MODE": "1",
                "QWEN3_ENV_CHECK_ROOT": str(fake_root),
                "QWEN3_ENV_CHECK_IMPORTS": "json,hashlib",
            }
        )

        result = subprocess.run(
            ["bash", str(ROOT / "tools" / "qwen3_env_check.sh")],
            cwd=ROOT,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("Qwen3-TTS environment check", result.stdout)
        self.assertIn("[PASS] production Python", result.stdout)
        self.assertIn("[PASS] Python import probe", result.stdout)


if __name__ == "__main__":
    unittest.main()
