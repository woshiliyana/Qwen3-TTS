# Qwen3-TTS Lightweight Production Control Layer Design

## Context

Qwen3-TTS now has a local long-form wrapper that keeps temp/cache/runtime state on the external drive, separates per-run workspaces from reusable voice caches, applies language presets, and writes segment health plus seam reports.

IndexTTS2 offers useful production lessons, but Qwen3-TTS should not copy its WebUI/service shape yet. IndexTTS2 is currently a Chinese/English service-style runtime with start/stop/status scripts, queue state, cooldown, retry/cancel, and external orchestration. Qwen3-TTS is being used as a multilingual CLI production runner for Spanish, Japanese, and French. The shared need is production safety around long-running local synthesis, not a shared language model interface.

## Goal

Add a lightweight production control layer for Qwen3-TTS before building heavier queue or repair tooling.

The first implementation should provide:

1. A preflight environment check.
2. A durable run registry.
3. A single-run lock with stale detection.

This should make production runs safer without changing the underlying synthesis path or introducing a daemon/WebUI.

## Non-Goals

- No WebUI, Gradio service wrapper, launchd service, or always-on daemon.
- No batch queue in this phase.
- No segment repair automation in this phase.
- No change to Qwen3 model loading, voice cloning, generation parameters, or audio quality heuristics.
- No attempt to unify Qwen3-TTS with IndexTTS2 APIs.

## Design Principles

- Keep Qwen3-TTS CLI-first.
- Keep all operational state under the external-drive project root.
- Prefer append-only evidence over hidden in-memory state.
- Make safety checks cheap enough to run before every production job.
- Preserve language-specific differences. Spanish, Japanese, and French may share orchestration code but not necessarily chunking heuristics or quality thresholds.

## Components

### 1. Environment Check

Add `tools/qwen3_env_check.sh`.

It should derive the repository root from its own location and verify:

- Project root is under `/Volumes`.
- Production Python exists at `.venv-qwen-prod/bin/python`.
- Model directory exists at `models/Qwen3-TTS-12Hz-1.7B-Base-local`.
- `ffmpeg` and `ffprobe` are available.
- Runtime directories exist or can be created:
  - `tmp/`
  - `tmp/gradio/`
  - `hf-cache/`
  - `.pip-cache-prod/`
  - `qwen3_tts_work/runs/`
  - `qwen3_tts_work/voices/`
- Python can import the minimum runtime libraries needed for local generation and quality checks.
- PyTorch reports whether MPS is available.
- Existing voice caches are summarized by invoking or reusing the same logic as `tools/qwen3_cache_status.sh`.

The environment check must stay lightweight. It must not load the Qwen3 model or synthesize audio. It may run import probes and `torch.backends.mps.is_available()`, but it should avoid any operation that triggers model checkpoint loading.

Output should be human-readable and return non-zero on hard blockers. Warnings may be allowed for missing optional voice caches.

Hard blockers:

- Repository root is not under `/Volumes`.
- Production Python is missing or not executable.
- Model directory is missing.
- `ffmpeg` or `ffprobe` is missing.
- Required runtime/cache directories cannot be created.
- Minimum Python imports fail.

Warnings:

- MPS is unavailable, because CPU fallback may be useful for diagnostics but should not be treated as production-ready.
- No reusable voice caches exist yet.
- A voice cache is incomplete.

### 2. Run Registry

Add append-only run records under:

```text
qwen3_tts_work/runs/index.jsonl
```

Each record should be one JSON object per line. The wrapper should append a record for both dry-run and real-run attempts.

Required fields for schema version 1:

- `schema_version`: integer, initially `1`.
- `event`: `planned`, `started`, `finished`, or `blocked`.
- `run_id`: stable generated id for this invocation.
- `recorded_at`: local timestamp for this registry event.
- `created_at`: local timestamp for the run invocation.
- `mode`: `dry_run` or `real_run`.
- `status`: `planned`, `running`, `completed`, `failed`, or `blocked`.
- `language`.
- `text_path`.
- `text_sha256`.
- `ref_audio`.
- `ref_audio_sha256`.
- `output_dir`.
- `output_name`.
- `work_dir`.
- `voice_cache_dir`.
- `max_chars`.
- `min_chars`.
- `max_new_tokens`.
- `pid`.
- `git_sha`: current `HEAD` SHA if available.
- `command_preview`: shell-escaped command string.
- `duration_s`: nullable until finalization.
- `exit_code`: nullable until finalization.
- `final_output`: nullable until known.
- `manifest_path`: nullable until known.
- `segment_health_report`: nullable until known.
- `quality_report`: nullable until known.
- `error`: nullable.

For real runs, append at least two records:

1. `event=started`, `status=running` before starting the Python runner.
2. `event=finished`, `status=completed` or `status=failed` after the runner exits.

For dry runs, append:

1. `event=planned`, `status=planned`.

For blocked runs, append:

1. `event=blocked`, `status=blocked`, with `error` containing the reason.
2. If blocked by an active lock, include a `lock_holder` object copied from the active lock payload.

Completed records should read actual output evidence from the run workspace when possible:

- `manifest_path`: `work_dir/manifest.json` when present.
- `final_output`: from the manifest `final_output` field when present, otherwise the expected output path.
- `segment_health_report`: from the manifest when present.
- `quality_report`: from the manifest when present.

This is append-only on purpose. Later tools can read the last record per `run_id` or per `work_dir` without rewriting history.

### 3. Single-Run Lock

Use an atomic lock directory under:

```text
qwen3_tts_work/run.lock.d/
```

The lock should be acquired by the shell wrapper before a real run starts and released on exit. Dry-run should never acquire the lock.

The lock must be acquired atomically with `mkdir`. The wrapper must not use a non-atomic "check then write" sequence. After acquiring the directory, it should write:

```text
qwen3_tts_work/run.lock.d/lock.json
```

Lock payload should be JSON:

- `run_id`.
- `pid`.
- `created_at`.
- `language`.
- `text_path`.
- `work_dir`.
- `voice_cache_dir`.

Stale detection:

- If the PID is alive, block the new real run and report the current lock holder.
- If the PID is not alive, treat the lock as stale, move the whole lock directory to `qwen3_tts_work/stale-locks/`, and continue.
- If the lock file is malformed, move it to `stale-locks/` with a timestamp suffix and continue.

Cleanup:

- The wrapper should release the lock on `EXIT`, `INT`, and `TERM` using a shell `trap`.
- If the process is killed with `kill -9`, stale detection on the next run is the recovery path.
- The wrapper should release only the lock whose `run_id` matches the current invocation.

This mirrors the IndexTTS2 lesson that local GPU/MPS inference should be treated as a single-lane resource.

### 4. Test Runner Seam

Add a test-only runner seam so verification does not load the model.

The shell wrapper should support:

- `QWEN3_RUNNER_BIN`: optional override for the Python runner executable or helper command used by tests.
- `QWEN3_SKIP_ENV_CHECK=1`: test-only opt-out for environment checks when a fake project root is used.
- `QWEN3_RECORD_DRY_RUN=0`: optional opt-out for recording dry-run registry rows during repeated manual path previews.

Default production behavior should not require these variables.

The fake runner used by tests should create a minimal `manifest.json` in `work_dir`, optionally create a dummy final output file, and exit with a configurable code. This lets tests verify registry finalization without model loading.

## Data Flow

### Dry Run

1. Resolve root, inputs, language preset, hashes, work dir, voice cache dir, and command.
2. Append a registry record with `mode=dry_run`, `event=planned`, and `status=planned`, unless `QWEN3_RECORD_DRY_RUN=0`.
3. Print resolved paths and command.
4. Exit without creating run directories beyond the registry parent if needed.
5. Do not acquire `run.lock.d`.

### Real Run

1. Resolve root, inputs, language preset, hashes, work dir, voice cache dir, and command.
2. Run environment checks that are cheap and local.
3. Acquire `run.lock.d`.
4. Append a registry record with `mode=real_run`, `event=started`, and `status=running`.
5. Run `tools/qwen3_longform_es.py`.
6. Capture exit code.
7. Append a final registry record with `event=finished` and `status=completed` or `status=failed`.
8. Release `run.lock.d`.

### Blocked Real Run

1. Resolve root, inputs, language preset, hashes, work dir, voice cache dir, and command.
2. If a valid active lock exists, append a registry record with `event=blocked`, `status=blocked`, and `lock_holder`.
3. Print the blocking run information.
4. Exit non-zero without mutating the active lock.

## Error Handling

- Missing Python, model directory, ffmpeg, ffprobe, or non-external root should block before generation.
- Active lock should block before generation, append a blocked registry record, and should not mutate the active lock.
- Stale or malformed lock should be preserved under `stale-locks/` before a new lock is acquired.
- Registry append failure should block real runs, because losing run evidence makes later recovery harder.
- If the Python runner fails, the wrapper should preserve the run workspace, append failure evidence, release the lock, and return the same non-zero exit code.
- If final registry append fails after the Python runner exits, the wrapper should still release the lock and return non-zero to force manual review.

## Testing

Add focused tests for:

- Environment check returns success against a temporary fake project root when required files/tools are stubbed or when the check supports a test mode.
- Dry-run appends a registry record but does not create or hold `run.lock.d`.
- `QWEN3_RECORD_DRY_RUN=0` skips dry-run registry writes.
- Real-run lock blocks a second active invocation and writes a `blocked` registry record.
- Stale lock is archived and replaced.
- Malformed lock is archived and replaced.
- Registry contains expected fields for dry-run and fake real-run paths.
- Fake real-run success reads final output and report paths from `manifest.json`.
- Fake real-run failure appends a failed final record and releases the lock.

The existing `tests/test_run_qwen3_longform.py` can be extended if the behavior stays shell-wrapper oriented. If the logic grows, move shared helpers into a small Python module and test it directly.

## Implementation Plan

### Phase 1: Shared Shell Helpers

- Add or inline helper functions for repo-root detection, SHA-256 hashing with `LC_ALL=C`, run id creation, JSONL append, git SHA lookup, and shell command preview.
- Keep helpers small enough to stay in `tools/run_qwen3_longform.sh` unless tests become too brittle.
- Add tests for dry-run registry rows and `QWEN3_RECORD_DRY_RUN=0`.

### Phase 2: Environment Check

- Add `tools/qwen3_env_check.sh`.
- Reuse `tools/qwen3_cache_status.sh` for voice cache summary.
- Add a test mode or fake-root mode so checks can be verified without requiring model loading.
- Wire real runs to call the check before locking.

### Phase 3: Atomic Lock

- Implement `run.lock.d` acquisition with `mkdir`.
- Write `lock.json` after acquiring the directory.
- Implement active-lock detection, stale lock archiving, malformed lock archiving, and trap cleanup.
- Add tests for active, stale, and malformed locks.

### Phase 4: Real-Run Registry Finalization

- Append `started` before invoking the runner.
- Invoke either the real Python runner or `QWEN3_RUNNER_BIN` in tests.
- Append `finished` after success or failure.
- Read `manifest.json` for actual final output and report paths.
- Preserve the Python runner exit code where possible.

### Phase 5: Documentation and Status Hooks

- Update `docs/local-longform-production.md` with env check, registry, lock behavior, and recovery instructions.
- Document that `qwen3_tts_work/runs/index.jsonl` is the source of run history.
- Defer `qwen3_runs_status.sh` to the next phase, but keep registry format sufficient for that future tool.

## Acceptance Criteria

- `tools/qwen3_env_check.sh` gives a clear pass/fail report on this Mac.
- `QWEN3_DRY_RUN=1 tools/run_qwen3_longform.sh ...` appends a dry-run registry record.
- A real run cannot start while an active `run.lock.d` exists.
- Active-lock blockage writes a `blocked` registry record.
- Stale locks do not permanently block production.
- The lock is acquired atomically and released on normal exits and interrupt/termination signals.
- Registry records are valid JSONL and enough to answer: what ran, with which voice, where output was expected, and how it ended.
- Registry records include `schema_version`, `event`, `recorded_at`, `git_sha`, and `command_preview`.
- Existing dry-run, cache status, Python compile, and shell syntax checks still pass.

## Implementation Defaults

The first implementation should keep these conservative defaults:

- Locking is enabled by default for real runs.
- Dry-run writes registry records but skips locking.
- Repeated manual dry-run previews can opt out with `QWEN3_RECORD_DRY_RUN=0`.
- Registry is append-only and not compacted.
- Segment repair is deferred to a later design after the control layer lands.
