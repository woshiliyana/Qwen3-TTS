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

LANGUAGE="$1"
REF_AUDIO="$2"
TEXT_PATH="$3"
OUTPUT_DIR="${4:-$(dirname "$TEXT_PATH")}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="$ROOT/models/Qwen3-TTS-12Hz-1.7B-Base-local"
PYTHON_BIN="$ROOT/.venv-qwen-prod/bin/python"
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

sha256_file() {
  local digest
  digest="$(env LC_ALL=C LANG=C shasum -a 256 "$1")"
  printf '%s\n' "${digest%% *}"
}

TEXT_BASE="$(basename "$TEXT_PATH")"
TEXT_STEM="${TEXT_BASE%.*}"
LANGUAGE_SAFE="$(printf '%s' "$LANGUAGE" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]_-')"
OUTPUT_NAME="${5:-${TEXT_STEM}_qwen3_${LANGUAGE_SAFE}_smooth.wav}"
VOICE_HASH="$(sha256_file "$REF_AUDIO")"
VOICE_HASH="${VOICE_HASH:0:12}"
TEXT_HASH="$(sha256_file "$TEXT_PATH")"
TEXT_HASH="${TEXT_HASH:0:12}"
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
VOICE_CACHE_DIR="$WORK_ROOT/voices/${LANGUAGE_SAFE}_${VOICE_HASH}"
WORK_DIR="$WORK_ROOT/runs/${TEXT_STEM}_${LANGUAGE_SAFE}_${TEXT_HASH}_${VOICE_HASH}_${MAX_CHARS}c_${MAX_NEW_TOKENS}tok"

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

if [[ "${QWEN3_DRY_RUN:-}" == "1" ]]; then
  printf 'work_dir=%s\n' "$WORK_DIR"
  printf 'voice_cache_dir=%s\n' "$VOICE_CACHE_DIR"
  printf 'output_dir=%s\n' "$OUTPUT_DIR"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$ROOT/tmp/gradio" "$ROOT/hf-cache" "$ROOT/.pip-cache-prod" "$OUTPUT_DIR" "$WORK_DIR" "$VOICE_CACHE_DIR"

export COPYFILE_DISABLE=1
export TMPDIR="$ROOT/tmp"
export GRADIO_TEMP_DIR="$ROOT/tmp/gradio"
export HF_HOME="$ROOT/hf-cache"
export PIP_CACHE_DIR="$ROOT/.pip-cache-prod"
export PYTHONPATH="$ROOT"

"${CMD[@]}"
