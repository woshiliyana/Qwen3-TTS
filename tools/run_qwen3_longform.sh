#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'USAGE'
Usage:
  tools/run_qwen3_longform.sh <Spanish|French|Japanese> <ref_audio.wav> <text.md> [output_dir] [output_name.wav]

Example:
  tools/run_qwen3_longform.sh Spanish \
    "/Volumes/woshiliyana/黑暗心理学频道/参考配音/年轻男声-西班牙语-在用.wav" \
    "/Volumes/woshiliyana/黑暗心理学频道/西班牙语频道/04/0427/psicologia-maquiavelica-bondad-oscura.md"
USAGE
  exit 2
fi

timestamp_local() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

archive_stamp() {
  date +"%Y%m%dT%H%M%S%z"
}

sha256_file() {
  local digest
  digest="$(env LC_ALL=C LANG=C shasum -a 256 "$1")"
  printf '%s\n' "${digest%% *}"
}

git_sha() {
  git -C "$ROOT" rev-parse HEAD 2>/dev/null || true
}

command_preview() {
  printf '%q ' "$@"
}

append_registry() {
  local event="$1"
  local status="$2"
  local mode="$3"
  local duration_s="${4:-}"
  local exit_code="${5:-}"
  local error="${6:-}"
  local lock_holder_path="${7:-}"

  mkdir -p "$RUNS_DIR"
  REGISTRY_PATH="$REGISTRY_PATH" \
  REGISTRY_EVENT="$event" \
  REGISTRY_RECORDED_AT="${RECORDED_AT:-$(timestamp_local)}" \
  REGISTRY_STATUS="$status" \
  REGISTRY_MODE="$mode" \
  REGISTRY_RUN_ID="$RUN_ID" \
  REGISTRY_CREATED_AT="$CREATED_AT" \
  REGISTRY_LANGUAGE="$LANGUAGE" \
  REGISTRY_TEXT_PATH="$TEXT_PATH" \
  REGISTRY_TEXT_SHA256="$TEXT_SHA256_FULL" \
  REGISTRY_REF_AUDIO="$REF_AUDIO" \
  REGISTRY_REF_AUDIO_SHA256="$REF_AUDIO_SHA256_FULL" \
  REGISTRY_OUTPUT_DIR="$OUTPUT_DIR" \
  REGISTRY_OUTPUT_NAME="$OUTPUT_NAME" \
  REGISTRY_WORK_DIR="$WORK_DIR" \
  REGISTRY_VOICE_CACHE_DIR="$VOICE_CACHE_DIR" \
  REGISTRY_MAX_CHARS="$MAX_CHARS" \
  REGISTRY_MIN_CHARS="$MIN_CHARS" \
  REGISTRY_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  REGISTRY_PID="$$" \
  REGISTRY_GIT_SHA="$GIT_SHA" \
  REGISTRY_COMMAND_PREVIEW="$COMMAND_PREVIEW" \
  REGISTRY_DURATION_S="$duration_s" \
  REGISTRY_EXIT_CODE="$exit_code" \
  REGISTRY_ERROR="$error" \
  REGISTRY_LOCK_HOLDER_PATH="$lock_holder_path" \
  "$REGISTRY_PYTHON" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path


def env(name: str) -> str:
    return os.environ.get(name, "")


def nullable_str(value: str) -> str | None:
    return value if value != "" else None


def nullable_int(value: str) -> int | None:
    return int(value) if value != "" else None


def nullable_float(value: str) -> float | None:
    return round(float(value), 3) if value != "" else None


work_dir = Path(env("REGISTRY_WORK_DIR"))
manifest_path = work_dir / "manifest.json"
manifest: dict[str, object] = {}
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

event = env("REGISTRY_EVENT")
status = env("REGISTRY_STATUS")
manifest_path_value: str | None = None
final_output: str | None = None
segment_health_report: str | None = None
quality_report: str | None = None
if event == "finished" and manifest:
    manifest_path_value = str(manifest_path)
    final_output = nullable_str(str(manifest.get("final_output") or ""))
    segment_health_report = nullable_str(str(manifest.get("segment_health_report") or ""))
    quality_report = nullable_str(str(manifest.get("quality_report") or ""))
elif event == "finished" and status == "completed":
    final_output = str(Path(env("REGISTRY_OUTPUT_DIR")) / env("REGISTRY_OUTPUT_NAME"))

record: dict[str, object] = {
    "schema_version": 1,
    "event": event,
    "run_id": env("REGISTRY_RUN_ID"),
    "recorded_at": env("RECORDED_AT") or env("REGISTRY_RECORDED_AT"),
    "created_at": env("REGISTRY_CREATED_AT"),
    "mode": env("REGISTRY_MODE"),
    "status": status,
    "language": env("REGISTRY_LANGUAGE"),
    "text_path": env("REGISTRY_TEXT_PATH"),
    "text_sha256": env("REGISTRY_TEXT_SHA256"),
    "ref_audio": env("REGISTRY_REF_AUDIO"),
    "ref_audio_sha256": env("REGISTRY_REF_AUDIO_SHA256"),
    "output_dir": env("REGISTRY_OUTPUT_DIR"),
    "output_name": env("REGISTRY_OUTPUT_NAME"),
    "work_dir": env("REGISTRY_WORK_DIR"),
    "voice_cache_dir": env("REGISTRY_VOICE_CACHE_DIR"),
    "max_chars": int(env("REGISTRY_MAX_CHARS")),
    "min_chars": int(env("REGISTRY_MIN_CHARS")),
    "max_new_tokens": int(env("REGISTRY_MAX_NEW_TOKENS")),
    "pid": int(env("REGISTRY_PID")),
    "git_sha": nullable_str(env("REGISTRY_GIT_SHA")),
    "command_preview": env("REGISTRY_COMMAND_PREVIEW").strip(),
    "duration_s": nullable_float(env("REGISTRY_DURATION_S")),
    "exit_code": nullable_int(env("REGISTRY_EXIT_CODE")),
    "final_output": final_output,
    "manifest_path": manifest_path_value,
    "segment_health_report": segment_health_report,
    "quality_report": quality_report,
    "error": nullable_str(env("REGISTRY_ERROR")),
}

lock_holder_path = nullable_str(env("REGISTRY_LOCK_HOLDER_PATH"))
if lock_holder_path:
    record["lock_holder"] = json.loads(Path(lock_holder_path).read_text(encoding="utf-8"))

registry_path = Path(env("REGISTRY_PATH"))
registry_path.parent.mkdir(parents=True, exist_ok=True)
with registry_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
PY
}

write_lock_payload() {
  LOCK_JSON="$LOCK_DIR/lock.json" \
  LOCK_RUN_ID="$RUN_ID" \
  LOCK_PID="$$" \
  LOCK_CREATED_AT="$CREATED_AT" \
  LOCK_LANGUAGE="$LANGUAGE" \
  LOCK_TEXT_PATH="$TEXT_PATH" \
  LOCK_WORK_DIR="$WORK_DIR" \
  LOCK_VOICE_CACHE_DIR="$VOICE_CACHE_DIR" \
  "$REGISTRY_PYTHON" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

path = Path(os.environ["LOCK_JSON"])
payload = {
    "run_id": os.environ["LOCK_RUN_ID"],
    "pid": int(os.environ["LOCK_PID"]),
    "created_at": os.environ["LOCK_CREATED_AT"],
    "language": os.environ["LOCK_LANGUAGE"],
    "text_path": os.environ["LOCK_TEXT_PATH"],
    "work_dir": os.environ["LOCK_WORK_DIR"],
    "voice_cache_dir": os.environ["LOCK_VOICE_CACHE_DIR"],
}
tmp_path = path.with_name("lock.json.tmp")
tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
os.replace(tmp_path, path)
PY
}

parse_lock_pid() {
  LOCK_JSON="$1" "$REGISTRY_PYTHON" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["LOCK_JSON"]).read_text(encoding="utf-8"))
pid = int(payload["pid"])
if pid <= 0:
    raise SystemExit(1)
print(pid)
PY
}

lock_matches_run() {
  LOCK_JSON="$1" EXPECTED_RUN_ID="$RUN_ID" "$REGISTRY_PYTHON" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["LOCK_JSON"]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("run_id") == os.environ["EXPECTED_RUN_ID"] else 1)
PY
}

archive_lock_dir() {
  local reason="$1"
  mkdir -p "$STALE_LOCKS_DIR"
  local dest="$STALE_LOCKS_DIR/run.lock.d.${reason}.$(archive_stamp).$$"
  while [[ -e "$dest" ]]; do
    dest="${dest}.${RANDOM}"
  done
  mv "$LOCK_DIR" "$dest"
  printf 'Archived %s lock: %s\n' "$reason" "$dest" >&2
}

acquire_lock() {
  mkdir -p "$WORK_ROOT"
  while true; do
    if mkdir "$LOCK_DIR" 2>/dev/null; then
      LOCK_ACQUIRED=1
      write_lock_payload
      return 0
    fi

    if [[ -f "$LOCK_DIR/lock.json" ]]; then
      local holder_pid
      if holder_pid="$(parse_lock_pid "$LOCK_DIR/lock.json")"; then
        if kill -0 "$holder_pid" 2>/dev/null; then
          LOCK_HOLDER_PATH="$LOCK_DIR/lock.json"
          return 2
        fi
        archive_lock_dir "stale"
        continue
      fi
      archive_lock_dir "malformed"
      continue
    fi

    archive_lock_dir "malformed"
  done
}

release_lock() {
  if [[ "${LOCK_ACQUIRED:-0}" != "1" ]]; then
    return 0
  fi
  if [[ -f "$LOCK_DIR/lock.json" ]] && lock_matches_run "$LOCK_DIR/lock.json"; then
    rm -f "$LOCK_DIR/lock.json"
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
  LOCK_ACQUIRED=0
}

cleanup_on_exit() {
  local status=$?
  release_lock
  exit "$status"
}

LANGUAGE="$1"
REF_AUDIO="$2"
TEXT_PATH="$3"
OUTPUT_DIR="${4:-$(dirname "$TEXT_PATH")}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="$ROOT/models/Qwen3-TTS-12Hz-1.7B-Base-local"
PYTHON_BIN="$ROOT/.venv-qwen-prod/bin/python"
REGISTRY_PYTHON="${QWEN3_REGISTRY_PYTHON:-python3}"

if [[ ! -f "$REF_AUDIO" ]]; then
  echo "Reference audio does not exist: $REF_AUDIO" >&2
  exit 1
fi
if [[ ! -f "$TEXT_PATH" ]]; then
  echo "Text file does not exist: $TEXT_PATH" >&2
  exit 1
fi
if [[ "$OUTPUT_DIR" != /Volumes/* ]]; then
  echo "Output directory must be on external storage under /Volumes: $OUTPUT_DIR" >&2
  exit 1
fi
if ! command -v "$REGISTRY_PYTHON" >/dev/null 2>&1; then
  echo "Registry Python is not available: $REGISTRY_PYTHON" >&2
  exit 1
fi

TEXT_BASE="$(basename "$TEXT_PATH")"
TEXT_STEM="${TEXT_BASE%.*}"
LANGUAGE_SAFE="$(printf '%s' "$LANGUAGE" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]_-')"
OUTPUT_NAME="${5:-${TEXT_STEM}_qwen3_${LANGUAGE_SAFE}_smooth.wav}"
REF_AUDIO_SHA256_FULL="$(sha256_file "$REF_AUDIO")"
VOICE_HASH="${REF_AUDIO_SHA256_FULL:0:12}"
TEXT_SHA256_FULL="$(sha256_file "$TEXT_PATH")"
TEXT_HASH="${TEXT_SHA256_FULL:0:12}"
case "$LANGUAGE" in
  Spanish)
    DEFAULT_MAX_CHARS=420
    DEFAULT_MIN_CHARS=220
    ;;
  French | Japanese)
    DEFAULT_MAX_CHARS=450
    DEFAULT_MIN_CHARS=250
    ;;
  *)
    echo "Unsupported language: $LANGUAGE" >&2
    exit 2
    ;;
esac
MAX_CHARS="${QWEN3_MAX_CHARS:-$DEFAULT_MAX_CHARS}"
MIN_CHARS="${QWEN3_MIN_CHARS:-$DEFAULT_MIN_CHARS}"
MAX_NEW_TOKENS="${QWEN3_MAX_NEW_TOKENS:-2048}"
MAX_SEGMENT_DURATION="${QWEN3_MAX_SEGMENT_DURATION:-120}"
BOUNDARY_PAUSE_MS="${QWEN3_BOUNDARY_PAUSE_MS:-180}"
EDGE_FADE_MS="${QWEN3_EDGE_FADE_MS:-8}"
WORK_ROOT="${QWEN3_WORK_ROOT:-$ROOT/qwen3_tts_work}"
RUNS_DIR="$WORK_ROOT/runs"
REGISTRY_PATH="$RUNS_DIR/index.jsonl"
LOCK_DIR="$WORK_ROOT/run.lock.d"
STALE_LOCKS_DIR="$WORK_ROOT/stale-locks"
VOICE_CACHE_DIR="$WORK_ROOT/voices/${LANGUAGE_SAFE}_${VOICE_HASH}"
WORK_DIR="$RUNS_DIR/${TEXT_STEM}_${LANGUAGE_SAFE}_${TEXT_HASH}_${VOICE_HASH}_${MAX_CHARS}c_${MAX_NEW_TOKENS}tok"
RUN_ID="${QWEN3_RUN_ID:-qwen3-$(date +%Y%m%dT%H%M%S%z)-$$-${RANDOM}}"
CREATED_AT="$(timestamp_local)"
GIT_SHA="$(git_sha)"
LOCK_ACQUIRED=0
LOCK_HOLDER_PATH=""

if [[ -n "${QWEN3_RUNNER_BIN:-}" ]]; then
  CMD=(
    "$QWEN3_RUNNER_BIN"
    --model-path "$MODEL_PATH" \
    --ref-audio "$REF_AUDIO" \
    --text-path "$TEXT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --work-dir "$WORK_DIR" \
    --voice-cache-dir "$VOICE_CACHE_DIR" \
    --output-name "$OUTPUT_NAME" \
    --language "$LANGUAGE" \
    --device mps \
    --dtype bfloat16 \
    --max-chars "$MAX_CHARS" \
    --min-chars "$MIN_CHARS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-segment-duration "$MAX_SEGMENT_DURATION" \
    --smooth-seams \
    --edge-fade-ms "$EDGE_FADE_MS" \
    --boundary-pause-ms "$BOUNDARY_PAUSE_MS"
  )
else
  CMD=(
    "$PYTHON_BIN" "$ROOT/tools/qwen3_longform_es.py"
    --model-path "$MODEL_PATH" \
    --ref-audio "$REF_AUDIO" \
    --text-path "$TEXT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --work-dir "$WORK_DIR" \
    --voice-cache-dir "$VOICE_CACHE_DIR" \
    --output-name "$OUTPUT_NAME" \
    --language "$LANGUAGE" \
    --device mps \
    --dtype bfloat16 \
    --max-chars "$MAX_CHARS" \
    --min-chars "$MIN_CHARS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-segment-duration "$MAX_SEGMENT_DURATION" \
    --smooth-seams \
    --edge-fade-ms "$EDGE_FADE_MS" \
    --boundary-pause-ms "$BOUNDARY_PAUSE_MS"
  )
fi
COMMAND_PREVIEW="$(command_preview "${CMD[@]}")"

export COPYFILE_DISABLE=1
export TMPDIR="$ROOT/tmp"
export GRADIO_TEMP_DIR="$ROOT/tmp/gradio"
export HF_HOME="$ROOT/hf-cache"
export PIP_CACHE_DIR="$ROOT/.pip-cache-prod"
export PYTHONPATH="$ROOT"

if [[ "${QWEN3_DRY_RUN:-}" == "1" ]]; then
  if [[ "${QWEN3_RECORD_DRY_RUN:-1}" != "0" ]]; then
    RECORDED_AT="$(timestamp_local)" append_registry "planned" "planned" "dry_run"
  fi
  printf 'work_dir=%s\n' "$WORK_DIR"
  printf 'voice_cache_dir=%s\n' "$VOICE_CACHE_DIR"
  printf 'output_dir=%s\n' "$OUTPUT_DIR"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ "${QWEN3_SKIP_ENV_CHECK:-}" != "1" ]]; then
  bash "$ROOT/tools/qwen3_env_check.sh"
fi

lock_status=0
acquire_lock || lock_status=$?
if [[ "$lock_status" -eq 2 ]]; then
  error_message="active Qwen3-TTS run lock is held"
  RECORDED_AT="$(timestamp_local)" append_registry "blocked" "blocked" "real_run" "" "" "$error_message" "$LOCK_HOLDER_PATH"
  echo "$error_message" >&2
  cat "$LOCK_HOLDER_PATH" >&2
  exit 1
elif [[ "$lock_status" -ne 0 ]]; then
  echo "Failed to acquire Qwen3-TTS run lock." >&2
  exit 1
fi

trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$ROOT/tmp/gradio" "$ROOT/hf-cache" "$ROOT/.pip-cache-prod" "$OUTPUT_DIR" "$WORK_DIR" "$VOICE_CACHE_DIR"

if ! RECORDED_AT="$(timestamp_local)" append_registry "started" "running" "real_run"; then
  echo "Failed to append started registry record." >&2
  exit 1
fi

run_started_at="$(date +%s)"
set +e
"${CMD[@]}"
runner_status=$?
set -e
run_finished_at="$(date +%s)"
duration_s=$((run_finished_at - run_started_at))

final_status="completed"
final_error=""
if [[ "$runner_status" -ne 0 ]]; then
  final_status="failed"
  final_error="runner exited with status $runner_status"
fi

if ! RECORDED_AT="$(timestamp_local)" append_registry "finished" "$final_status" "real_run" "$duration_s" "$runner_status" "$final_error"; then
  echo "Failed to append final registry record." >&2
  exit 1
fi

release_lock
exit "$runner_status"
