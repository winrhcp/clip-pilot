from __future__ import annotations

import os
from pathlib import Path

import requests


def synthesize_thai_voice(text: str, output_audio: Path) -> None:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("TTS_MODEL", "tts-1")
    voice = os.getenv("TTS_VOICE", "alloy")
    if not api_key:
        raise RuntimeError("Missing API key for dubbing. Set LLM_API_KEY or OPENAI_API_KEY.")

    cleaned = " ".join(text.split())
    if not cleaned:
        raise RuntimeError("Cannot dub empty text.")

    payload = {
        "model": model,
        "voice": voice,
        "input": cleaned,
        "format": "mp3",
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        r = requests.post(f"{base_url.rstrip('/')}/audio/speech", headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        output_audio.write_bytes(r.content)
    except requests.HTTPError as exc:
        body = ""
        try:
            body = exc.response.text[:500]
        except Exception:
            body = ""
        raise RuntimeError(f"TTS HTTP error: {exc}. Response: {body}") from exc
    except Exception as exc:
        raise RuntimeError(f"TTS request failed: {exc}") from exc
