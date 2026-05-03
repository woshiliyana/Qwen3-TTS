#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


PURE_SEPARATOR_LINE_PATTERN = re.compile(r"^\s*([-*_])\1{2,}\s*$")
SUPPORTED_LANGUAGES = ("Spanish", "French", "Japanese")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate long-form multilingual TTS with Qwen3-TTS Base.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--text-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--voice-cache-dir",
        default=None,
        help="Directory for reusable voice prompt and normalized reference caches. Defaults to --work-dir.",
    )
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--language", default="Spanish", choices=SUPPORTED_LANGUAGES)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-chars", type=int, default=450)
    parser.add_argument("--min-chars", type=int, default=250)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-segment-duration", type=float, default=120.0)
    parser.add_argument("--min-retry-chars", type=int, default=120)
    parser.add_argument(
        "--smooth-seams",
        action="store_true",
        help="Apply short edge fades and add micro-pauses at short segment boundaries during final concat.",
    )
    parser.add_argument("--edge-fade-ms", type=float, default=8.0)
    parser.add_argument("--boundary-pause-ms", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--smoke", action="store_true", help="Generate only the first segment and skip concat.")
    parser.add_argument("--skip-quality-gate", action="store_true", help="Downgrade quality gate failures to warnings instead of blocking output.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing segment files.")
    return parser.parse_args()


def configure_logging(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "run.log"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def clean_markdown_text(markdown_text: str) -> str:
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_front_matter(text)
    text = _strip_fenced_code_blocks(text)
    text = _strip_html_images(text)
    text = _strip_markdown_images(text)
    text = _strip_reference_link_definitions(text)
    text = _strip_markdown_links(text)
    text = _strip_raw_urls(text)
    text = _strip_table_separator_rows(text)
    text = _strip_separators(text)
    text = _strip_markdown_markers(text)
    return _normalize_whitespace(text)


def _strip_front_matter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :]
    if text.startswith("+++\n"):
        end = text.find("\n+++\n", 4)
        if end != -1:
            return text[end + 5 :]
    return text


def _strip_fenced_code_blocks(text: str) -> str:
    return re.sub(r"(^|\n)(```|~~~).*?(\n```|\n~~~)", "\n", text, flags=re.DOTALL)


def _strip_html_images(text: str) -> str:
    return re.sub(r"<img\b[^>]*>", " ", text, flags=re.IGNORECASE)


def _strip_markdown_images(text: str) -> str:
    return re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)


def _strip_reference_link_definitions(text: str) -> str:
    return re.sub(r"^\s*\[[^\]]+\]:\s+\S+.*$", "", text, flags=re.MULTILINE)


def _strip_markdown_links(text: str) -> str:
    return re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)


def _strip_raw_urls(text: str) -> str:
    text = re.sub(r"<https?://[^>]+>", " ", text)
    return re.sub(r"https?://\S+", " ", text)


def _strip_table_separator_rows(text: str) -> str:
    return re.sub(r"^\s*\|?[\s:-]+\|[\s|:-]*$", "", text, flags=re.MULTILINE)


def _strip_separators(text: str) -> str:
    return "\n".join(
        raw_line for raw_line in text.split("\n") if not PURE_SEPARATOR_LINE_PATTERN.fullmatch(raw_line)
    ).strip()


def _strip_markdown_markers(text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</?(?:p|div|span|section|article|strong|em|code|blockquote|ul|ol|li|h[1-6])[^>]*>",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return text.replace("|", " ")


def _normalize_whitespace(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            cleaned_lines.append(line)
    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_spanish_text(text: str, max_chars: int, min_chars: int) -> list[str]:
    sentence_parts = re.split(r"(?<=[.!?¿¡;:。！？；：])\s*", text.replace("\n", " "))
    chunks: list[str] = []
    current = ""
    for part in sentence_parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_sentence(part, max_chars))
            continue
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate) <= max_chars or len(current) < min_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            current = part
    if current.strip():
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def _split_long_sentence(text: str, max_chars: int) -> list[str]:
    clauses = re.split(r"(?<=[,，、])\s*", text)
    chunks: list[str] = []
    current = ""
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        candidate = f"{current} {clause}".strip() if current else clause
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = clause
            while len(current) > max_chars:
                cut = current.rfind(" ", 0, max_chars)
                if cut < max_chars // 2:
                    cut = max_chars
                chunks.append(current[:cut].strip())
                current = current[cut:].strip()
    if current:
        chunks.append(current)
    return chunks


def split_for_retry(text: str) -> tuple[str, str]:
    midpoint = len(text) // 2
    candidates: list[int] = []
    for pattern in [r"[.!?¿¡;:。！？；：]\s*", r"[,，、]\s*", r"\s+"]:
        for match in re.finditer(pattern, text):
            candidates.append(match.end())
    if not candidates:
        return text[:midpoint].strip(), text[midpoint:].strip()
    cut = min(candidates, key=lambda pos: abs(pos - midpoint))
    left = text[:cut].strip()
    right = text[cut:].strip()
    if not left or not right:
        return text[:midpoint].strip(), text[midpoint:].strip()
    return left, right


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def is_external_volume_path(path: str | Path) -> bool:
    resolved = Path(path).expanduser().resolve(strict=False)
    return len(resolved.parts) >= 3 and resolved.parts[1] == "Volumes"


def require_external_volume_path(name: str, path: str | Path | None) -> None:
    if path is None or str(path).strip() == "":
        raise RuntimeError(f"{name} must be set and point to external storage.")
    if not is_external_volume_path(path):
        raise RuntimeError(f"{name} must point to external storage under /Volumes, got: {path!r}")


def load_runtime_dependencies() -> None:
    global librosa, np, sf, torch, Qwen3TTSModel, VoiceClonePromptItem

    import librosa as _librosa
    import numpy as _np
    import soundfile as _sf
    import torch as _torch
    from qwen_tts import Qwen3TTSModel as _Qwen3TTSModel
    from qwen_tts import VoiceClonePromptItem as _VoiceClonePromptItem

    librosa = _librosa
    np = _np
    sf = _sf
    torch = _torch
    Qwen3TTSModel = _Qwen3TTSModel
    VoiceClonePromptItem = _VoiceClonePromptItem


def normalize_reference_audio(ref_audio: Path, work_dir: Path, ref_audio_sha256: str) -> Path:
    out_path = work_dir / "reference_24k_mono.wav"
    meta_path = work_dir / "reference_24k_mono.meta.json"
    if out_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if meta.get("ref_audio_sha256") == ref_audio_sha256:
            return out_path
        logging.warning("Reference audio cache does not match current voice; regenerating.")
    elif out_path.exists():
        logging.warning("Reference audio cache has no metadata; regenerating.")

    if out_path.exists():
        out_path.unlink()
    wav, _sr = librosa.load(str(ref_audio), sr=24000, mono=True)
    sf.write(out_path, wav.astype(np.float32), 24000, subtype="PCM_16")
    meta_path.write_text(
        json.dumps(
            {
                "source": str(ref_audio),
                "ref_audio_sha256": ref_audio_sha256,
                "sample_rate": 24000,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def load_or_create_prompt(
    tts: Qwen3TTSModel,
    ref_audio: Path,
    prompt_path: Path,
    ref_audio_sha256: str,
) -> list[VoiceClonePromptItem]:
    meta_path = prompt_path.with_suffix(".meta.json")
    if prompt_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if meta.get("ref_audio_sha256") == ref_audio_sha256:
            logging.info("Loading voice prompt: %s", prompt_path)
            return load_prompt(prompt_path)
        logging.warning("Voice prompt cache does not match current voice; regenerating.")
    elif prompt_path.exists():
        logging.warning("Voice prompt cache has no metadata; regenerating.")

    logging.info("Creating x-vector-only voice prompt from: %s", ref_audio)
    items = tts.create_voice_clone_prompt(ref_audio=str(ref_audio), ref_text=None, x_vector_only_mode=True)
    save_prompt(items, prompt_path)
    meta_path.write_text(
        json.dumps(
            {
                "source": str(ref_audio),
                "ref_audio_sha256": ref_audio_sha256,
                "x_vector_only_mode": True,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return items


def normalize_optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def normalize_optional_int(value: int | None) -> int | None:
    return None if value is None else int(value)


def build_generation_signature(args: argparse.Namespace, model_path: Path, ref_audio_sha256: str) -> dict[str, Any]:
    return {
        "language": args.language,
        "model_path": str(model_path),
        "ref_audio_sha256": ref_audio_sha256,
        "max_chars": int(args.max_chars),
        "min_chars": int(args.min_chars),
        "max_new_tokens": int(args.max_new_tokens),
        "max_segment_duration": float(args.max_segment_duration),
        "dtype": args.dtype,
        "temperature": normalize_optional_float(args.temperature),
        "top_p": normalize_optional_float(args.top_p),
        "top_k": normalize_optional_int(args.top_k),
        "repetition_penalty": normalize_optional_float(args.repetition_penalty),
    }


def save_prompt(items: list[VoiceClonePromptItem], prompt_path: Path) -> None:
    payload = {"items": [asdict(item) for item in items]}
    torch.save(payload, prompt_path)


def load_prompt(prompt_path: Path) -> list[VoiceClonePromptItem]:
    payload = torch.load(prompt_path, map_location="cpu", weights_only=True)
    items: list[VoiceClonePromptItem] = []
    for item in payload["items"]:
        ref_code = item.get("ref_code")
        ref_spk = item.get("ref_spk_embedding")
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(item.get("x_vector_only_mode", True)),
                icl_mode=bool(item.get("icl_mode", False)),
                ref_text=item.get("ref_text"),
            )
        )
    return items


def build_manifest(
    text_path: Path,
    cleaned_text: str,
    chunks: list[str],
    args: argparse.Namespace,
    model_path: Path,
    ref_audio: Path,
    ref_audio_sha256: str,
) -> dict[str, Any]:
    generation_signature = build_generation_signature(args, model_path, ref_audio_sha256)
    return {
        "source": str(text_path),
        "source_sha256": sha256_text(text_path.read_text(encoding="utf-8")),
        "cleaned_sha256": sha256_text(cleaned_text),
        "ref_audio": str(ref_audio),
        "ref_audio_sha256": ref_audio_sha256,
        **generation_signature,
        "generation_signature": generation_signature,
        "segments": [
            {
                "index": i,
                "text": chunk,
                "chars": len(chunk),
                "sha256": sha256_text(chunk),
                "status": "pending",
                "attempts": 0,
                "path": None,
                "duration": None,
                "seconds": None,
                "error": None,
            }
            for i, chunk in enumerate(chunks, start=1)
        ],
    }


def load_or_reset_manifest(
    manifest_path: Path,
    text_path: Path,
    cleaned_text: str,
    chunks: list[str],
    args: argparse.Namespace,
    model_path: Path,
    ref_audio: Path,
    ref_audio_sha256: str,
) -> dict[str, Any]:
    current_signature = build_generation_signature(args, model_path, ref_audio_sha256)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_segment_hashes = [str(item.get("sha256") or "") for item in existing.get("segments", [])]
        current_segment_hashes = [sha256_text(chunk) for chunk in chunks]
        manifest_matches = (
            existing.get("cleaned_sha256") == sha256_text(cleaned_text)
            and existing.get("generation_signature") == current_signature
            and existing_segment_hashes == current_segment_hashes
        )
        if manifest_matches:
            return existing
        logging.warning("Manifest does not match current text/chunks; rebuilding manifest.")
    return build_manifest(text_path, cleaned_text, chunks, args, model_path, ref_audio, ref_audio_sha256)


def persist_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def wav_duration(path: Path) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def generate_segment(
    tts: Qwen3TTSModel,
    prompt_items: list[VoiceClonePromptItem],
    text: str,
    language: str,
    out_path: Path,
    gen_kwargs: dict[str, Any],
) -> float:
    wavs, sr = tts.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=prompt_items,
        **gen_kwargs,
    )
    wav = np.asarray(wavs[0], dtype=np.float32)
    sf.write(out_path, wav, sr, subtype="PCM_16")
    return wav_duration(out_path)


def unique_bad_path(path: Path, reason: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}.{reason}.{stamp}{path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.{reason}.{stamp}_{counter}{path.suffix}")
        counter += 1
    return candidate


def generate_segment_guarded(
    tts: Qwen3TTSModel,
    prompt_items: list[VoiceClonePromptItem],
    text: str,
    language: str,
    out_path: Path,
    gen_kwargs: dict[str, Any],
    retry_dir: Path,
    max_segment_duration: float,
    min_retry_chars: int,
    depth: int = 0,
) -> float:
    duration = generate_segment(
        tts=tts,
        prompt_items=prompt_items,
        text=text,
        language=language,
        out_path=out_path,
        gen_kwargs=gen_kwargs,
    )
    if duration <= max_segment_duration:
        return duration

    bad_path = unique_bad_path(out_path, f"overlong_d{depth}")
    out_path.replace(bad_path)
    logging.warning(
        "Segment output was overlong: %.2fs > %.2fs; moved to %s",
        duration,
        max_segment_duration,
        bad_path,
    )

    if len(text) <= min_retry_chars or depth >= 4:
        raise RuntimeError(
            f"Segment remained overlong ({duration:.2f}s) and cannot be split safely "
            f"(chars={len(text)}, depth={depth})."
        )

    left, right = split_for_retry(text)
    if not left or not right:
        raise RuntimeError(f"Failed to split overlong segment safely (chars={len(text)}).")

    retry_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    for part_index, part_text in enumerate([left, right], start=1):
        part_path = retry_dir / f"{out_path.stem}_d{depth}_{part_index}.wav"
        logging.info(
            "Retrying overlong segment part %s.%s chars=%s",
            out_path.stem,
            part_index,
            len(part_text),
        )
        generate_segment_guarded(
            tts=tts,
            prompt_items=prompt_items,
            text=part_text,
            language=language,
            out_path=part_path,
            gen_kwargs=gen_kwargs,
            retry_dir=retry_dir,
            max_segment_duration=max_segment_duration,
            min_retry_chars=min_retry_chars,
            depth=depth + 1,
        )
        part_paths.append(part_path)

    concat_segments(part_paths, retry_dir / f"{out_path.stem}_d{depth}_concat.txt", out_path)
    merged_duration = wav_duration(out_path)
    if merged_duration > max_segment_duration:
        bad_merged_path = unique_bad_path(out_path, f"merged_overlong_d{depth}")
        out_path.replace(bad_merged_path)
        raise RuntimeError(
            f"Merged retry segment is still overlong: {merged_duration:.2f}s > "
            f"{max_segment_duration:.2f}s; moved to {bad_merged_path}"
        )
    logging.info("Merged retry parts into %s duration=%.2fs", out_path, merged_duration)
    return merged_duration


def concat_segments(segment_paths: list[Path], concat_path: Path, final_path: Path) -> None:
    concat_lines = [f"file '{escape_ffmpeg_concat_path(path)}'" for path in segment_paths]
    concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-c:a",
        "pcm_s16le",
        str(final_path),
    ]
    subprocess.run(cmd, check=True)


def escape_ffmpeg_concat_path(path: Path) -> str:
    value = path.as_posix()
    if "\n" in value or "\r" in value:
        raise ValueError(f"ffmpeg concat path cannot contain newlines: {value!r}")
    return value.replace("'", r"'\''")


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=False)
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, int(sr)


def rms_db(wav: np.ndarray) -> float:
    if len(wav) == 0:
        return -120.0
    value = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64)))
    if value <= 1e-12:
        return -120.0
    return 20.0 * math.log10(value)


def peak(wav: np.ndarray) -> float:
    if len(wav) == 0:
        return 0.0
    return float(np.max(np.abs(wav)))


def edge_silence_seconds(wav: np.ndarray, sr: int, side: str, threshold_db: float = -45.0) -> float:
    frame = max(1, int(sr * 0.02))
    threshold = 10 ** (threshold_db / 20)
    samples = wav[::-1] if side == "end" else wav
    silent = 0
    for offset in range(0, len(samples), frame):
        chunk = samples[offset : offset + frame]
        if len(chunk) == 0:
            break
        value = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
        if value < threshold:
            silent += len(chunk)
        else:
            break
    return silent / sr


def analyze_segment_seams(segment_paths: list[Path], report_path: Path) -> dict[str, Any]:
    loaded: list[tuple[Path, np.ndarray, int]] = []
    for path in segment_paths:
        wav, sr = read_wav_mono(path)
        loaded.append((path, wav, sr))

    sample_rates = sorted({sr for _, _, sr in loaded})
    if len(sample_rates) != 1:
        raise RuntimeError(f"Mixed segment sample rates: {sample_rates}")
    sr = sample_rates[0]
    edge_samples = max(1, int(sr * 0.2))
    seams: list[dict[str, Any]] = []

    for index in range(len(loaded) - 1):
        left_path, left, _ = loaded[index]
        right_path, right, _ = loaded[index + 1]
        tail = left[-edge_samples:]
        head = right[:edge_samples]
        jump = float(abs(left[-1] - right[0])) if len(left) and len(right) else 0.0
        tail_silence = edge_silence_seconds(left, sr, "end")
        head_silence = edge_silence_seconds(right, sr, "start")
        pause = tail_silence + head_silence

        risk = 0
        flags: list[str] = []
        if jump > 0.06:
            risk += 3
            flags.append("sample_jump")
        if pause < 0.08:
            risk += 2
            flags.append("very_short_pause")
        if pause > 0.90:
            risk += 1
            flags.append("long_pause")
        if peak(tail[-max(1, int(sr * 0.02)) :]) > 0.18 and tail_silence < 0.03:
            risk += 1
            flags.append("hot_tail")
        if peak(head[: max(1, int(sr * 0.02))]) > 0.18 and head_silence < 0.03:
            risk += 1
            flags.append("hot_head")

        seams.append(
            {
                "between": [index + 1, index + 2],
                "left": str(left_path),
                "right": str(right_path),
                "sample_jump": round(jump, 6),
                "tail_rms_db_200ms": round(rms_db(tail), 2),
                "head_rms_db_200ms": round(rms_db(head), 2),
                "tail_peak_200ms": round(peak(tail), 6),
                "head_peak_200ms": round(peak(head), 6),
                "tail_silence_s": round(tail_silence, 3),
                "head_silence_s": round(head_silence, 3),
                "estimated_pause_s": round(pause, 3),
                "risk_score": risk,
                "flags": flags,
            }
        )

    durations = [len(wav) / sr for _, wav, _ in loaded]
    pauses = [item["estimated_pause_s"] for item in seams] or [0.0]
    jumps = [item["sample_jump"] for item in seams] or [0.0]
    flag_counts: dict[str, int] = {}
    for item in seams:
        for flag in item["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    report = {
        "sample_rate": sr,
        "segments_total": len(segment_paths),
        "duration_total_s": round(sum(durations), 3),
        "duration_min_s": round(min(durations), 3),
        "duration_median_s": round(statistics.median(durations), 3),
        "duration_max_s": round(max(durations), 3),
        "seams_total": len(seams),
        "risk_counts": {
            str(score): sum(1 for item in seams if item["risk_score"] == score)
            for score in sorted({item["risk_score"] for item in seams} or {0})
        },
        "flag_counts": flag_counts,
        "sample_jump_max": round(max(jumps), 6),
        "sample_jump_p95": round(float(np.percentile(jumps, 95)), 6),
        "pause_min_s": round(min(pauses), 3),
        "pause_median_s": round(statistics.median(pauses), 3),
        "pause_max_s": round(max(pauses), 3),
        "top_risk_seams": sorted(seams, key=lambda item: (item["risk_score"], item["sample_jump"]), reverse=True)[:15],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def evaluate_quality_gate(
    segment_health: dict[str, Any],
    seam_quality: dict[str, Any],
    gate_path: Path,
    *,
    max_risk3_ratio: float = 0.05,
) -> dict[str, Any]:
    reasons: list[str] = []

    if segment_health["hard_clipping_segments"]:
        reasons.append(
            f"hard clipping in segment(s): {segment_health['hard_clipping_segments']}"
        )

    risk_counts = seam_quality.get("risk_counts", {})
    seams_total = seam_quality.get("seams_total", 0)

    high_risk_count = sum(
        count for score_str, count in risk_counts.items() if int(score_str) >= 4
    )
    if high_risk_count:
        reasons.append(f"{high_risk_count} seam(s) with risk >= 4")

    risk3_count = int(risk_counts.get("3", 0))
    risk3_ratio = risk3_count / seams_total if seams_total else 0.0
    if risk3_ratio > max_risk3_ratio:
        reasons.append(
            f"risk=3 seams: {risk3_count}/{seams_total} ({risk3_ratio:.1%}) exceeds {max_risk3_ratio:.0%} threshold"
        )

    passed = len(reasons) == 0
    warnings: list[str] = []
    if passed and risk3_count > 0:
        warnings.append(
            f"risk=3 seams: {risk3_count}/{seams_total} ({risk3_ratio:.1%}) within threshold"
        )

    gate = {
        "passed": passed,
        "reasons": reasons,
        "warnings": warnings,
        "risk_counts": risk_counts,
        "seams_total": seams_total,
        "hard_clipping_segments": segment_health["hard_clipping_segments"],
        "suspect_count": segment_health["suspect_count"],
        "max_risk3_ratio": max_risk3_ratio,
    }
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    return gate


def median_or_none(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return float(statistics.median(clean))


def rounded_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def estimate_head_f0_median(wav: np.ndarray, sr: int, seconds: float = 12.0) -> float | None:
    frames = min(len(wav), int(sr * seconds))
    if frames < sr:
        return None
    head = wav[:frames]
    try:
        f0 = librosa.yin(head, fmin=60, fmax=350, sr=sr, frame_length=2048, hop_length=512)
    except Exception as exc:
        logging.warning("Failed to estimate segment head f0: %s", exc)
        return None
    clean = [float(value) for value in f0 if math.isfinite(float(value))]
    return median_or_none(clean)


def analyze_segment_health(
    segment_paths: list[Path],
    manifest_segments: list[dict[str, Any]],
    report_path: Path,
) -> dict[str, Any]:
    if len(segment_paths) != len(manifest_segments):
        raise RuntimeError(
            f"Segment health scan needs matching paths and manifest rows: "
            f"{len(segment_paths)} paths vs {len(manifest_segments)} manifest rows"
        )

    rows: list[dict[str, Any]] = []
    sample_rates: set[int] = set()
    for path, item in zip(segment_paths, manifest_segments):
        wav, sr = read_wav_mono(path)
        sample_rates.add(sr)
        duration = len(wav) / sr if sr else 0.0
        chars = int(item.get("chars") or len(str(item.get("text") or "")) or 0)
        rms = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64))) if len(wav) else 0.0
        abs_wav = np.abs(wav) if len(wav) else np.array([], dtype=np.float32)
        peak_value = float(np.max(abs_wav)) if len(abs_wav) else 0.0
        p95 = float(np.percentile(abs_wav, 95)) if len(abs_wav) else 0.0
        clip_ratio = float(np.mean(abs_wav >= 0.98)) if len(abs_wav) else 0.0
        f0_median = estimate_head_f0_median(wav, sr)
        seconds_per_100_chars = duration / max(chars, 1) * 100.0
        rows.append(
            {
                "index": int(item["index"]),
                "path": str(path),
                "duration_s": duration,
                "chars": chars,
                "seconds_per_100_chars": seconds_per_100_chars,
                "rms": rms,
                "peak": peak_value,
                "p95_abs": p95,
                "clip_ratio": clip_ratio,
                "head_f0_median_hz": f0_median,
                "text_preview": str(item.get("text") or "")[:180],
            }
        )

    if len(sample_rates) != 1:
        raise RuntimeError(f"Mixed segment sample rates during health scan: {sorted(sample_rates)}")
    sample_rate = next(iter(sample_rates)) if sample_rates else None

    medians = {
        "duration_s": median_or_none([row["duration_s"] for row in rows]),
        "seconds_per_100_chars": median_or_none([row["seconds_per_100_chars"] for row in rows]),
        "rms": median_or_none([row["rms"] for row in rows]),
        "peak": median_or_none([row["peak"] for row in rows]),
        "p95_abs": median_or_none([row["p95_abs"] for row in rows]),
        "head_f0_median_hz": median_or_none(
            [
                row["head_f0_median_hz"]
                for row in rows
                if row["head_f0_median_hz"] is not None
            ]
        ),
    }
    thresholds = {
        "clip_peak_min": 0.98,
        "clip_ratio_min": 0.0,
        "rms_high_multiplier": 1.55,
        "p95_high_multiplier": 1.55,
        "f0_outlier_hz_delta": 28.0,
        "pace_slow_multiplier": 1.65,
        "pace_fast_multiplier": 0.55,
        "duration_high_multiplier": 1.8,
    }

    suspect_segments: list[dict[str, Any]] = []
    hard_clipping_segments: list[int] = []
    for row in rows:
        flags: list[str] = []
        if row["peak"] >= thresholds["clip_peak_min"] or row["clip_ratio"] > thresholds["clip_ratio_min"]:
            flags.append("clipping")
            hard_clipping_segments.append(row["index"])
        if medians["rms"] and row["rms"] > medians["rms"] * thresholds["rms_high_multiplier"]:
            flags.append("rms_high")
        if medians["p95_abs"] and row["p95_abs"] > medians["p95_abs"] * thresholds["p95_high_multiplier"]:
            flags.append("p95_high")
        if row["head_f0_median_hz"] is not None and medians["head_f0_median_hz"] is not None:
            if abs(row["head_f0_median_hz"] - medians["head_f0_median_hz"]) > thresholds["f0_outlier_hz_delta"]:
                flags.append("f0_outlier")
        if medians["seconds_per_100_chars"]:
            if row["seconds_per_100_chars"] > medians["seconds_per_100_chars"] * thresholds["pace_slow_multiplier"]:
                flags.append("pace_slow")
            if row["seconds_per_100_chars"] < medians["seconds_per_100_chars"] * thresholds["pace_fast_multiplier"]:
                flags.append("pace_fast")
        if medians["duration_s"] and row["duration_s"] > medians["duration_s"] * thresholds["duration_high_multiplier"]:
            flags.append("duration_high")

        row["flags"] = flags
        if flags:
            suspect_segments.append(row)

    flag_counts: dict[str, int] = {}
    for row in suspect_segments:
        for flag in row["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    def public_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            **row,
            "duration_s": rounded_or_none(row["duration_s"]),
            "seconds_per_100_chars": rounded_or_none(row["seconds_per_100_chars"], 2),
            "rms": rounded_or_none(row["rms"], 4),
            "peak": rounded_or_none(row["peak"], 4),
            "p95_abs": rounded_or_none(row["p95_abs"], 4),
            "clip_ratio": rounded_or_none(row["clip_ratio"], 6),
            "head_f0_median_hz": rounded_or_none(row["head_f0_median_hz"], 1),
        }

    report = {
        "sample_rate": sample_rate,
        "segments_total": len(rows),
        "medians": {key: rounded_or_none(value, 4) for key, value in medians.items()},
        "thresholds": thresholds,
        "hard_clipping_segments": hard_clipping_segments,
        "suspect_count": len(suspect_segments),
        "flag_counts": flag_counts,
        "suspect_segments": [public_row(row) for row in suspect_segments],
        "segments": [public_row(row) for row in rows],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def apply_edge_fades(wav: np.ndarray, sr: int, fade_ms: float, fade_in: bool, fade_out: bool) -> np.ndarray:
    if fade_ms <= 0:
        return wav
    fade_samples = min(len(wav) // 2, max(1, int(sr * fade_ms / 1000)))
    out = wav.copy()
    if fade_in and fade_samples > 0:
        out[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    if fade_out and fade_samples > 0:
        out[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    return out


def concat_segments_smoothed(
    segment_paths: list[Path],
    final_path: Path,
    edge_fade_ms: float,
    boundary_pause_ms: float,
) -> dict[str, Any]:
    loaded = [(*read_wav_mono(path), path) for path in segment_paths]
    sample_rates = sorted({sr for _, sr, _ in loaded})
    if len(sample_rates) != 1:
        raise RuntimeError(f"Mixed segment sample rates: {sample_rates}")
    sr = sample_rates[0]
    target_pause = max(0.0, boundary_pause_ms / 1000.0)
    pieces: list[np.ndarray] = []
    added_pauses: list[float] = []

    for index, (wav, _, _) in enumerate(loaded):
        pieces.append(
            apply_edge_fades(
                wav,
                sr,
                edge_fade_ms,
                fade_in=index > 0,
                fade_out=index < len(loaded) - 1,
            )
        )
        if index < len(loaded) - 1:
            next_wav = loaded[index + 1][0]
            existing_pause = edge_silence_seconds(wav, sr, "end") + edge_silence_seconds(next_wav, sr, "start")
            add_pause = max(0.0, target_pause - existing_pause)
            added_pauses.append(add_pause)
            if add_pause > 0:
                pieces.append(np.zeros(int(sr * add_pause), dtype=np.float32))

    final = np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)
    sf.write(str(final_path), final, sr, subtype="PCM_16")
    return {
        "sample_rate": sr,
        "edge_fade_ms": edge_fade_ms,
        "boundary_pause_ms": boundary_pause_ms,
        "added_boundaries": sum(1 for value in added_pauses if value > 0),
        "added_pause_total_s": round(sum(added_pauses), 3),
        "added_pause_max_s": round(max(added_pauses) if added_pauses else 0.0, 3),
    }


def main() -> int:
    args = parse_args()
    require_external_volume_path("TMPDIR", os.environ.get("TMPDIR"))
    require_external_volume_path("GRADIO_TEMP_DIR", os.environ.get("GRADIO_TEMP_DIR"))
    require_external_volume_path("HF_HOME", os.environ.get("HF_HOME"))
    require_external_volume_path("PIP_CACHE_DIR", os.environ.get("PIP_CACHE_DIR"))

    model_path = Path(args.model_path).expanduser().resolve(strict=False)
    ref_audio = Path(args.ref_audio).expanduser().resolve(strict=False)
    text_path = Path(args.text_path).expanduser().resolve(strict=False)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    work_dir = Path(args.work_dir).expanduser().resolve(strict=False)
    voice_cache_dir = (
        Path(args.voice_cache_dir).expanduser().resolve(strict=False) if args.voice_cache_dir else work_dir
    )
    require_external_volume_path("model_path", model_path)
    require_external_volume_path("output_dir", output_dir)
    require_external_volume_path("work_dir", work_dir)
    require_external_volume_path("voice_cache_dir", voice_cache_dir)
    if not model_path.is_dir():
        raise RuntimeError(f"Model directory does not exist: {model_path}")
    if not ref_audio.is_file():
        raise RuntimeError(f"Reference audio does not exist: {ref_audio}")
    if not text_path.is_file():
        raise RuntimeError(f"Text file does not exist: {text_path}")
    segments_dir = work_dir / "segments"
    retry_dir = work_dir / "retry_parts"
    final_path = output_dir / args.output_name
    manifest_path = work_dir / "manifest.json"
    cleaned_text_path = work_dir / "cleaned_text.txt"
    concat_path = work_dir / "concat.txt"

    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    voice_cache_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(work_dir)

    logging.info("TMPDIR=%s", os.environ.get("TMPDIR"))
    logging.info("GRADIO_TEMP_DIR=%s", os.environ.get("GRADIO_TEMP_DIR"))
    logging.info("HF_HOME=%s", os.environ.get("HF_HOME"))
    logging.info("model=%s", model_path)
    logging.info("ref_audio=%s", ref_audio)
    logging.info("text=%s", text_path)
    logging.info("work_dir=%s", work_dir)
    logging.info("voice_cache_dir=%s", voice_cache_dir)

    cleaned_text = clean_markdown_text(text_path.read_text(encoding="utf-8"))
    if not cleaned_text:
        raise RuntimeError(f"Text file is empty after Markdown cleanup: {text_path}")
    cleaned_text_path.write_text(cleaned_text, encoding="utf-8")
    chunks = split_spanish_text(cleaned_text, max_chars=args.max_chars, min_chars=args.min_chars)
    if args.smoke:
        chunks = chunks[:1]
    if not chunks:
        raise RuntimeError(f"Text produced no generation segments: {text_path}")
    logging.info("Cleaned chars=%s segments=%s smoke=%s", len(cleaned_text), len(chunks), args.smoke)

    ref_audio_sha256 = sha256_file(ref_audio)
    manifest = load_or_reset_manifest(
        manifest_path,
        text_path,
        cleaned_text,
        chunks,
        args,
        model_path,
        ref_audio,
        ref_audio_sha256,
    )
    persist_manifest(manifest_path, manifest)

    load_runtime_dependencies()

    reference_24k = normalize_reference_audio(ref_audio, voice_cache_dir, ref_audio_sha256)
    logging.info("reference_24k=%s", reference_24k)

    logging.info("Loading Qwen3-TTS model...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map=args.device,
        dtype=torch_dtype(args.dtype),
        attn_implementation=None,
    )
    logging.info("Model loaded in %.2fs", time.time() - t0)
    prompt_path = voice_cache_dir / "voice_clone_prompt.pt"
    prompt_items = load_or_create_prompt(tts, reference_24k, prompt_path, ref_audio_sha256)

    gen_kwargs: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        gen_kwargs["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty

    completed_paths: list[Path] = []
    for item in manifest["segments"]:
        index = int(item["index"])
        out_path = segments_dir / f"{index:04d}.wav"
        can_reuse_existing = (
            item.get("status") == "completed"
            and item.get("path") == str(out_path)
            and item.get("sha256") == sha256_text(str(item.get("text") or ""))
        )
        if out_path.exists() and not args.force and can_reuse_existing:
            try:
                duration = wav_duration(out_path)
                if duration > args.max_segment_duration:
                    bad_path = unique_bad_path(out_path, "existing_overlong")
                    out_path.replace(bad_path)
                    logging.warning(
                        "Existing segment %04d is overlong: %.2fs > %.2fs; moved to %s",
                        index,
                        duration,
                        args.max_segment_duration,
                        bad_path,
                    )
                else:
                    item.update(status="completed", path=str(out_path), duration=duration, error=None)
                    completed_paths.append(out_path)
                    logging.info("Skipping existing segment %04d duration=%.2fs", index, duration)
                    continue
            except Exception:
                logging.warning("Existing segment %04d is unreadable; regenerating.", index)

        logging.info("Generating segment %04d/%04d chars=%s", index, len(manifest["segments"]), item["chars"])
        item["attempts"] = int(item.get("attempts") or 0) + 1
        started = time.time()
        try:
            duration = generate_segment_guarded(
                tts=tts,
                prompt_items=prompt_items,
                text=item["text"],
                language=args.language,
                out_path=out_path,
                gen_kwargs=gen_kwargs,
                retry_dir=retry_dir,
                max_segment_duration=args.max_segment_duration,
                min_retry_chars=args.min_retry_chars,
            )
            elapsed = time.time() - started
            item.update(
                status="completed",
                path=str(out_path),
                duration=duration,
                seconds=elapsed,
                error=None,
            )
            completed_paths.append(out_path)
            logging.info("Completed segment %04d audio=%.2fs elapsed=%.2fs", index, duration, elapsed)
        except Exception as exc:
            item.update(status="failed", path=str(out_path), error=f"{type(exc).__name__}: {exc}")
            persist_manifest(manifest_path, manifest)
            logging.exception("Failed segment %04d", index)
            return 2
        persist_manifest(manifest_path, manifest)

    if args.smoke:
        logging.info("Smoke run complete; skipping concat.")
        return 0

    if len(completed_paths) != len(manifest["segments"]):
        logging.error("Not all segments completed: %s/%s", len(completed_paths), len(manifest["segments"]))
        return 3

    segment_health_report_path = work_dir / "segment_health_report.json"
    segment_health_report = analyze_segment_health(
        completed_paths,
        manifest["segments"],
        segment_health_report_path,
    )
    logging.info(
        "Segment health report: suspects=%s hard_clipping=%s path=%s",
        segment_health_report["suspect_count"],
        segment_health_report["hard_clipping_segments"],
        segment_health_report_path,
    )
    if segment_health_report["suspect_count"]:
        logging.warning(
            "Segment health scan found %s suspect segment(s): %s",
            segment_health_report["suspect_count"],
            [
                {
                    "index": item["index"],
                    "flags": item["flags"],
                    "duration_s": item["duration_s"],
                    "peak": item["peak"],
                    "head_f0_median_hz": item["head_f0_median_hz"],
                }
                for item in segment_health_report["suspect_segments"]
            ],
        )
    manifest["segment_health_report"] = str(segment_health_report_path)

    quality_report_name = "pre_smoothing_quality_report.json" if args.smooth_seams else "quality_report.json"
    quality_report_path = work_dir / quality_report_name
    quality_report = analyze_segment_seams(completed_paths, quality_report_path)
    logging.info(
        "Quality report: seams=%s risk_counts=%s max_jump=%s pause_min=%ss path=%s",
        quality_report["seams_total"],
        quality_report["risk_counts"],
        quality_report["sample_jump_max"],
        quality_report["pause_min_s"],
        quality_report_path,
    )
    manifest["quality_report"] = str(quality_report_path)

    gate_path = work_dir / "quality_gate.json"
    gate = evaluate_quality_gate(segment_health_report, quality_report, gate_path)
    manifest["quality_gate"] = str(gate_path)
    persist_manifest(manifest_path, manifest)
    if gate["passed"]:
        for warning in gate.get("warnings", []):
            logging.warning("Quality gate warning: %s", warning)
        logging.info("Quality gate passed path=%s", gate_path)
    else:
        for reason in gate["reasons"]:
            logging.error("Quality gate FAILED: %s", reason)
        if args.skip_quality_gate:
            logging.warning("Quality gate failed but --skip-quality-gate is set; proceeding anyway")
        else:
            logging.error("Output not produced. Review reports in %s or re-run with --skip-quality-gate", work_dir)
            return 4

    logging.info("Concatenating %s segments -> %s", len(completed_paths), final_path)
    if args.smooth_seams:
        smoothing_report = concat_segments_smoothed(
            completed_paths,
            final_path,
            edge_fade_ms=args.edge_fade_ms,
            boundary_pause_ms=args.boundary_pause_ms,
        )
        logging.info("Seam smoothing: %s", smoothing_report)
        manifest["seam_smoothing"] = smoothing_report
    else:
        concat_segments(completed_paths, concat_path, final_path)
    final_duration = wav_duration(final_path)
    manifest["final_output"] = str(final_path)
    manifest["final_duration"] = final_duration
    persist_manifest(manifest_path, manifest)
    logging.info("Final output duration=%.2fs path=%s", final_duration, final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
