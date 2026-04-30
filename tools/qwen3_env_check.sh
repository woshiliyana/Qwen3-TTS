#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ "${QWEN3_ENV_CHECK_TEST_MODE:-}" == "1" && -n "${QWEN3_ENV_CHECK_ROOT:-}" ]]; then
  ROOT="$QWEN3_ENV_CHECK_ROOT"
fi

WORK_ROOT="${QWEN3_WORK_ROOT:-$ROOT/qwen3_tts_work}"
PYTHON_BIN="${QWEN3_ENV_CHECK_PYTHON:-$ROOT/.venv-qwen-prod/bin/python}"
MODEL_PATH="${QWEN3_ENV_CHECK_MODEL_PATH:-$ROOT/models/Qwen3-TTS-12Hz-1.7B-Base-local}"
IMPORTS="${QWEN3_ENV_CHECK_IMPORTS:-torch,transformers,librosa,soundfile,sox,numpy,qwen_tts}"

errors=0
warnings=0

pass() {
  printf '[PASS] %s\n' "$1"
}

fail() {
  printf '[FAIL] %s\n' "$1"
  errors=$((errors + 1))
}

warn() {
  printf '[WARN] %s\n' "$1"
  warnings=$((warnings + 1))
}

printf 'Qwen3-TTS environment check\n'
printf 'root=%s\n' "$ROOT"

if [[ "$ROOT" == /Volumes/* ]]; then
  pass "project root is on external storage"
else
  fail "project root must be under /Volumes: $ROOT"
fi

if [[ -x "$PYTHON_BIN" ]]; then
  pass "production Python: $PYTHON_BIN"
else
  fail "production Python missing or not executable: $PYTHON_BIN"
fi

if [[ -d "$MODEL_PATH" ]]; then
  pass "model directory: $MODEL_PATH"
else
  fail "model directory missing: $MODEL_PATH"
fi

for binary in ffmpeg ffprobe; do
  if command -v "$binary" >/dev/null 2>&1; then
    pass "$binary available: $(command -v "$binary")"
  else
    fail "$binary is not available on PATH"
  fi
done

runtime_dirs=(
  "$ROOT/tmp"
  "$ROOT/tmp/gradio"
  "$ROOT/hf-cache"
  "$ROOT/.pip-cache-prod"
  "$WORK_ROOT/runs"
  "$WORK_ROOT/voices"
)

for dir in "${runtime_dirs[@]}"; do
  if mkdir -p "$dir" 2>/dev/null; then
    pass "runtime directory ready: $dir"
  else
    fail "runtime directory cannot be created: $dir"
  fi
done

if [[ -x "$PYTHON_BIN" ]]; then
  import_output="$(
    PYTHONPATH="$ROOT" "$PYTHON_BIN" - "$IMPORTS" <<'PY' 2>&1
from __future__ import annotations

import importlib
import sys

modules = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
loaded = {}
failed = []
for module in modules:
    try:
        loaded[module] = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report exact import blocker.
        failed.append(f"{module}: {type(exc).__name__}: {exc}")

if failed:
    print("IMPORT_FAILED")
    for item in failed:
        print(item)
    raise SystemExit(1)

print("IMPORT_OK " + ",".join(modules))
torch = loaded.get("torch")
if torch is None:
    print("MPS_SKIPPED torch import not requested")
else:
    available = bool(torch.backends.mps.is_available())
    print(f"MPS_AVAILABLE {str(available).lower()}")
PY
  )"
  import_status=$?
  if [[ "$import_status" -eq 0 ]]; then
    pass "Python import probe: $IMPORTS"
    if printf '%s\n' "$import_output" | grep -q '^MPS_AVAILABLE true$'; then
      pass "PyTorch MPS available"
    elif printf '%s\n' "$import_output" | grep -q '^MPS_AVAILABLE false$'; then
      warn "PyTorch MPS unavailable; CPU fallback is diagnostic-only for production"
    else
      warn "MPS check skipped"
    fi
  else
    fail "Python import probe failed"
    printf '%s\n' "$import_output"
  fi
fi

printf '\nVoice cache status\n'
cache_output="$(QWEN3_WORK_ROOT="$WORK_ROOT" bash "$SCRIPT_DIR/qwen3_cache_status.sh" 2>&1)"
cache_status=$?
if [[ "$cache_status" -ne 0 ]]; then
  warn "voice cache status command failed"
  printf '%s\n' "$cache_output"
else
  printf '%s\n' "$cache_output"
  cache_rows="$(printf '%s\n' "$cache_output" | awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }')"
  if [[ "$cache_rows" -eq 0 ]]; then
    warn "no reusable voice caches found"
  elif printf '%s\n' "$cache_output" | awk 'NR > 1 && /missing/ { found=1 } END { exit found ? 0 : 1 }'; then
    warn "one or more voice caches are incomplete"
  else
    pass "voice caches are complete"
  fi
fi

printf '\nSummary: errors=%s warnings=%s\n' "$errors" "$warnings"
if [[ "$errors" -ne 0 ]]; then
  exit 1
fi
exit 0
