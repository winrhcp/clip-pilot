from __future__ import annotations

import json
import os
from typing import List

import requests

from ai_worker.whisper.transcriber import Segment


def translate_segments_to_thai(segments: List[Segment]) -> List[Segment]:
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        return segments

    texts = [seg.text.strip() for seg in segments]
    if not any(texts):
        return segments

    prompt = (
        "Translate each subtitle line to natural Thai.\n"
        "Keep line count and order exactly the same.\n"
        "Return strict JSON: {\"translations\":[\"...\", \"...\"]}\n"
        "If a line is already Thai, keep meaning and improve only when needed."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": json.dumps({"lines": texts}, ensure_ascii=False)},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        r = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        translated = parsed.get("translations", [])
        if not isinstance(translated, list) or len(translated) != len(segments):
            return segments
    except Exception:
        return segments

    out: List[Segment] = []
    for seg, txt in zip(segments, translated):
        text = txt.strip() if isinstance(txt, str) else seg.text
        out.append(Segment(start=seg.start, end=seg.end, text=text or seg.text))
    return out
