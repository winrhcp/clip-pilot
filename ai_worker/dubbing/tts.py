from __future__ import annotations

import os
from pathlib import Path

import requests


def synthesize_thai_voice(text: str, output_audio: Path) -> bool:
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
    voice = os.getenv("TTS_VOICE", "alloy")
    if not api_key:
        return False

    cleaned = " ".join(text.split())
    if not cleaned:
        return False

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
    except Exception:
        return False
    return True
