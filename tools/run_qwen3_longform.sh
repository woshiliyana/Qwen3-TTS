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

ROOT="/Volumes/My Passport/vibe coding/Qwen3-TTS"
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
TEXT_BASE="$(basename "$TEXT_PATH")"
TEXT_STEM="${TEXT_BASE%.*}"
LANGUAGE_SAFE="$(printf '%s' "$LANGUAGE" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]_-')"
OUTPUT_NAME="${5:-${TEXT_STEM}_qwen3_${LANGUAGE_SAFE}_smooth.wav}"
VOICE_HASH="$(shasum -a 256 "$REF_AUDIO")"
VOICE_HASH="${VOICE_HASH%% *}"
VOICE_HASH="${VOICE_HASH:0:12}"
MAX_CHARS="${QWEN3_MAX_CHARS:-450}"
MIN_CHARS="${QWEN3_MIN_CHARS:-250}"
MAX_NEW_TOKENS="${QWEN3_MAX_NEW_TOKENS:-2048}"
MAX_SEGMENT_DURATION="${QWEN3_MAX_SEGMENT_DURATION:-120}"
BOUNDARY_PAUSE_MS="${QWEN3_BOUNDARY_PAUSE_MS:-180}"
EDGE_FADE_MS="${QWEN3_EDGE_FADE_MS:-8}"
WORK_DIR="$OUTPUT_DIR/qwen3_tts_work_${TEXT_STEM}_${LANGUAGE_SAFE}_${VOICE_HASH}_${MAX_CHARS}c_${MAX_NEW_TOKENS}tok"

mkdir -p "$ROOT/tmp/gradio" "$ROOT/hf-cache" "$ROOT/.pip-cache-prod" "$OUTPUT_DIR" "$WORK_DIR"

export COPYFILE_DISABLE=1
export TMPDIR="$ROOT/tmp"
export GRADIO_TEMP_DIR="$ROOT/tmp/gradio"
export HF_HOME="$ROOT/hf-cache"
export PIP_CACHE_DIR="$ROOT/.pip-cache-prod"
export PYTHONPATH="$ROOT"

"$PYTHON_BIN" "$ROOT/tools/qwen3_longform_es.py" \
  --model-path "$MODEL_PATH" \
  --ref-audio "$REF_AUDIO" \
  --text-path "$TEXT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --work-dir "$WORK_DIR" \
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
