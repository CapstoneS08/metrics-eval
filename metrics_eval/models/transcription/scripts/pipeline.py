"""
Transcription pipeline: model-agnostic interface.

Backends supported:
- openai_whisper: OpenAI audio transcription API (default)
- faster_whisper: local faster-whisper (optional; requires pip install faster-whisper)

Returns:
{
  "transcript": str,
  "latency_s": float,
  "backend": str,
  "model": str
}
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional


def _transcribe_openai_whisper(audio_path: Path, model: str = "whisper-1") -> Dict:
    import os
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Load from .env or env var.")

    client = OpenAI(api_key=api_key)

    t0 = time.time()
    with audio_path.open("rb") as f:
        # Current SDK typically expects: client.audio.transcriptions.create(model=..., file=...)
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
        )
    latency = time.time() - t0

    # resp may be str-like or have .text depending on SDK version
    text = getattr(resp, "text", None)
    if text is None:
        text = str(resp)

    return {
        "transcript": text,
        "latency_s": latency,
        "backend": "openai_whisper",
        "model": model,
    }


_FASTER_WHISPER_MODEL = None
_FASTER_WHISPER_MODEL_NAME = None


def _get_faster_whisper_model(model_name: str, device: str = "cpu", compute_type: str = "int8"):
    global _FASTER_WHISPER_MODEL, _FASTER_WHISPER_MODEL_NAME
    if _FASTER_WHISPER_MODEL is None or _FASTER_WHISPER_MODEL_NAME != model_name:
        from faster_whisper import WhisperModel
        _FASTER_WHISPER_MODEL = WhisperModel(model_name, device=device, compute_type=compute_type)
        _FASTER_WHISPER_MODEL_NAME = model_name
    return _FASTER_WHISPER_MODEL


def _transcribe_faster_whisper(
    audio_path: Path,
    model: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = "en",
) -> Dict:
    t0 = time.time()
    fw = _get_faster_whisper_model(model, device=device, compute_type=compute_type)
    segments, info = fw.transcribe(str(audio_path), language=language)
    text = "".join(seg.text for seg in segments).strip()
    latency = time.time() - t0

    return {
        "transcript": text,
        "latency_s": latency,
        "backend": "faster_whisper",
        "model": model,
    }


def transcribe(
    audio_path: str | Path,
    backend: str = "openai_whisper",
    model: str = "whisper-1",
    **kwargs,
) -> Dict:
    """
    Main entrypoint.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    backend = backend.strip().lower()

    if backend == "openai_whisper":
        return _transcribe_openai_whisper(audio_path=audio_path, model=model)

    if backend == "faster_whisper":
        return _transcribe_faster_whisper(audio_path=audio_path, model=model, **kwargs)

    raise ValueError(f"Unknown backend: {backend}. Use openai_whisper or faster_whisper.")