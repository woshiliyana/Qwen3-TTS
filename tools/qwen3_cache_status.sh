#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_ROOT="${QWEN3_WORK_ROOT:-$ROOT/qwen3_tts_work}"
VOICES_DIR="$WORK_ROOT/voices"

status_file() {
  if [[ -f "$1" ]]; then
    printf 'ok'
  else
    printf 'missing'
  fi
}

printf 'language\tvoice_hash\tprompt\treference\tprompt_meta\treference_meta\tpath\n'
if [[ ! -d "$VOICES_DIR" ]]; then
  exit 0
fi

shopt -s nullglob
for voice_dir in "$VOICES_DIR"/*; do
  [[ -d "$voice_dir" ]] || continue
  voice_name="$(basename "$voice_dir")"
  language="${voice_name%%_*}"
  voice_hash="${voice_name#*_}"
  if [[ "$voice_hash" == "$voice_name" ]]; then
    voice_hash=""
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$language" \
    "$voice_hash" \
    "$(status_file "$voice_dir/voice_clone_prompt.pt")" \
    "$(status_file "$voice_dir/reference_24k_mono.wav")" \
    "$(status_file "$voice_dir/voice_clone_prompt.meta.json")" \
    "$(status_file "$voice_dir/reference_24k_mono.meta.json")" \
    "$voice_dir"
done
