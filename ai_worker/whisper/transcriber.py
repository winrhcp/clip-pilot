from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel


@dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe(video_path: Path, model_name: str = "medium") -> List[Segment]:
    model = WhisperModel(model_name, device="auto", compute_type="int8")
    raw_segments, _info = model.transcribe(str(video_path), vad_filter=True)
    segments: List[Segment] = []
    for s in raw_segments:
        text = (s.text or "").strip()
        if not text:
            continue
        segments.append(Segment(start=float(s.start), end=float(s.end), text=text))
    return segments
