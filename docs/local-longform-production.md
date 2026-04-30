# Local Long-Form Qwen3-TTS Production

This repository has a local long-form wrapper for external-drive production:

```bash
tools/run_qwen3_longform.sh <Spanish|French|Japanese> <reference-audio> <text.md> [output-dir] [output-name.wav]
```

The wrapper keeps runtime state on external storage:

- `TMPDIR=/Volumes/My Passport/vibe coding/Qwen3-TTS/tmp`
- `GRADIO_TEMP_DIR=/Volumes/My Passport/vibe coding/Qwen3-TTS/tmp/gradio`
- `HF_HOME=/Volumes/My Passport/vibe coding/Qwen3-TTS/hf-cache`
- `PIP_CACHE_DIR=/Volumes/My Passport/vibe coding/Qwen3-TTS/.pip-cache-prod`
- Per-script workspaces: `/Volumes/My Passport/vibe coding/Qwen3-TTS/qwen3_tts_work/runs/`
- Reusable voice caches: `/Volumes/My Passport/vibe coding/Qwen3-TTS/qwen3_tts_work/voices/`

The wrapper derives the project root from its own location, so it can be run from any working directory after the repository is moved. The output directory must also be under `/Volumes`. Final WAVs still go next to the script by default, but generation state no longer lives in the script folder.

Each run workspace is keyed by script stem, language, script hash, reference-voice hash, chunk size, and token budget. Each voice cache is keyed by language and reference-voice hash, so Spanish and Japanese prompt caches stay separate while repeated runs with the same reference audio can reuse:

- `voice_clone_prompt.pt`
- `voice_clone_prompt.meta.json`
- `reference_24k_mono.wav`
- `reference_24k_mono.meta.json`

To preview the resolved paths without generating audio:

```bash
QWEN3_DRY_RUN=1 tools/run_qwen3_longform.sh Spanish /Volumes/.../voice.wav /Volumes/.../script.md
```

Dry-run previews append a planned row to the local run registry by default. For repeated path previews where no registry evidence is useful, opt out explicitly:

```bash
QWEN3_DRY_RUN=1 QWEN3_RECORD_DRY_RUN=0 tools/run_qwen3_longform.sh Spanish /Volumes/.../voice.wav /Volumes/.../script.md
```

Before a real run, check the local production environment:

```bash
tools/qwen3_env_check.sh
```

The check verifies the external-drive root, production Python, local model directory, `ffmpeg`/`ffprobe`, runtime/cache directories, lightweight Python imports, PyTorch MPS availability, and reusable voice-cache status. It does not load the Qwen3 model or synthesize audio.

To inspect reusable voice caches:

```bash
tools/qwen3_cache_status.sh
```

The status output is tab-separated:

```text
language  voice_hash  prompt  reference  prompt_meta  reference_meta  path
```

## Run Registry and Lock

Run history is append-only at:

```text
qwen3_tts_work/runs/index.jsonl
```

Each row is one JSON object with schema version `1`. Dry runs write `event=planned`; real runs write `event=started` before invoking the runner and `event=finished` after it exits. Blocked real runs write `event=blocked`, including the active `lock_holder` when another run owns the single-run lock.

The registry records the run id, local timestamps, mode/status, input and reference hashes, output/work/cache paths, chunk/token settings, current git SHA, command preview, duration, exit code, and final evidence paths from `manifest.json` when available.

Real runs use a single-lane atomic lock:

```text
qwen3_tts_work/run.lock.d/
```

The wrapper acquires the lock with `mkdir` and writes `run.lock.d/lock.json` only after acquisition. Dry runs never acquire this lock. If a second real run starts while the lock holder PID is alive, the wrapper exits non-zero and records a blocked registry row without changing the active lock.

If the lock holder PID is gone, or if `lock.json` is malformed, the wrapper moves the entire lock directory under:

```text
qwen3_tts_work/stale-locks/
```

Then it acquires a fresh lock and continues. Normal exits, interrupts, and termination signals release only the lock whose `run_id` matches the current invocation. If a process is killed with `kill -9`, the next run's stale-lock detection is the recovery path.

Testing can replace the real model runner with a lightweight executable:

```bash
QWEN3_SKIP_ENV_CHECK=1 QWEN3_RUNNER_BIN=/path/to/fake-runner tools/run_qwen3_longform.sh Spanish /Volumes/.../voice.wav /Volumes/.../script.md
```

`QWEN3_RUNNER_BIN` is intended for test helpers that accept the same CLI flags and write a minimal `manifest.json`; production runs should leave it unset.

## Default Quality Checks

Every full run now writes two quality reports before final closeout:

- `segment_health_report.json`
- `pre_smoothing_quality_report.json` when `--smooth-seams` is used, otherwise `quality_report.json`

`segment_health_report.json` is the main guard for bad generated segments. It scans every segment WAV and records:

- `hard_clipping_segments`: segments whose peak reaches full scale or has clipped samples.
- `suspect_count`: number of segments with outlier flags.
- `suspect_segments`: the exact segment numbers, metrics, flags, and text previews.
- Per-segment metrics: duration, RMS, peak, p95 absolute amplitude, first-12-second median F0, and seconds per 100 characters.

The scan is designed to catch failures like:

- A segment suddenly becoming much louder than neighboring segments.
- A male voice segment jumping into a higher, female-like register.
- A generated segment speaking much slower or faster than the rest.
- Hard clipping or near-clipping.

If `suspect_count > 0` or `hard_clipping_segments` is non-empty, do not treat the audio as final. Regenerate those exact segment numbers, replace the segment WAVs, then rerun the scan and final concat.

`pre_smoothing_quality_report.json` is the seam report. It checks adjacent segment boundaries for:

- sample jumps
- very short pauses
- long pauses
- hot tails or hot heads

This report is useful for deciding which boundaries to spot-listen, but it does not replace the segment health scan.

## Required Closeout

For every production output:

1. Confirm `segment_health_report.json` exists.
2. Confirm `hard_clipping_segments` is empty.
3. Confirm `suspect_count` is `0`, or document and repair every suspect segment.
4. Confirm the seam report exists.
5. Confirm the final WAV with `ffprobe`:

```bash
ffprobe -v error \
  -show_entries format=duration,size:stream=codec_name,sample_rate,channels \
  -of default=noprint_wrappers=1 \
  "/Volumes/.../final.wav"
```

6. Spot-listen the top seam risks and any repaired segment boundaries.

## Local Repair Pattern

When a segment is bad:

1. Map the problem time to the segment using cumulative durations from `manifest.json`.
2. Generate 2-3 candidates for only that segment.
3. Compare candidate metrics to neighboring segments.
4. Pick the candidate with no clipping, normal RMS/p95, stable F0, and normal pace.
5. Back up the original bad segment using a descriptive suffix.
6. Replace the segment WAV and update `manifest.json` with a `repair` block.
7. Rerun `segment_health_report.json`, seam analysis, and final concat.

Do not regenerate the whole long-form audio unless multiple unrelated segments fail or the reference voice itself is wrong.

## Known Production Defaults

Spanish long-form has been stable around:

```bash
QWEN3_MAX_CHARS=420
QWEN3_MIN_CHARS=220
QWEN3_MAX_NEW_TOKENS=2048
```

Japanese long-form has been stable around:

```bash
QWEN3_MAX_CHARS=450
QWEN3_MIN_CHARS=250
QWEN3_MAX_NEW_TOKENS=2048
```

The wrapper now applies these language presets by default:

- `Spanish`: `QWEN3_MAX_CHARS=420`, `QWEN3_MIN_CHARS=220`
- `Japanese`: `QWEN3_MAX_CHARS=450`, `QWEN3_MIN_CHARS=250`
- `French`: `QWEN3_MAX_CHARS=450`, `QWEN3_MIN_CHARS=250`

Explicit environment variables still override the preset.

Longer chunks can work, but they increase the chance of slow, loud, high-register, or overlong segment failures.
