from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import List

import requests
from ai_worker.whisper.transcriber import Segment


KEYWORDS = {
    "wow", "wait", "important", "profit", "shock", "omg", "must", "secret",
    "?????", "??????????", "?????", "????", "????"
}


@dataclass
class ClipCandidate:
    start: float
    end: float
    score: float
    title: str
    reason: str


def _score_text(text: str) -> float:
    low = text.lower()
    keyword_hits = sum(1 for k in KEYWORDS if k in low)
    exclam = text.count("!")
    qmarks = text.count("?")
    length_bonus = min(len(text) / 160.0, 1.0) * 0.2
    base = (keyword_hits * 0.22) + (exclam * 0.08) + (qmarks * 0.05) + length_bonus
    return min(base, 1.0)


def score_highlights(segments: List[Segment], clip_len_sec: int = 45, top_k: int = 5) -> List[ClipCandidate]:
    candidates: List[ClipCandidate] = []
    for seg in segments:
        score = _score_text(seg.text)
        if score < 0.25:
            continue
        end = min(seg.start + clip_len_sec, seg.end + clip_len_sec)
        title = seg.text[:60] + ("..." if len(seg.text) > 60 else "")
        candidates.append(
            ClipCandidate(
                start=max(seg.start - 2.0, 0.0),
                end=end,
                score=score,
                title=title,
                reason="keyword+punctuation heuristic",
            )
        )

    candidates.sort(key=lambda x: x.score, reverse=True)

    selected: List[ClipCandidate] = []
    for c in candidates:
        overlap = any(not (c.end <= s.start or c.start >= s.end) for s in selected)
        if not overlap:
            selected.append(c)
        if len(selected) >= top_k:
            break
    return selected


def serialize_candidates(candidates: List[ClipCandidate]) -> List[dict]:
    return [asdict(c) for c in candidates]


def _build_candidate_text(segments: List[Segment], cand: ClipCandidate, idx: int) -> str:
    txt_parts: List[str] = []
    for seg in segments:
        if seg.end < cand.start or seg.start > cand.end:
            continue
        txt_parts.append(seg.text.strip())
    txt = " ".join(x for x in txt_parts if x)
    txt = txt[:700]
    return f"{idx}. [{cand.start:.2f}-{cand.end:.2f}] {txt}"


def rerank_with_llm(segments: List[Segment], candidates: List[ClipCandidate], top_k: int) -> List[ClipCandidate]:
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        return candidates[:top_k]

    shortlist = candidates[: min(len(candidates), max(top_k * 4, 12))]
    lines = [_build_candidate_text(segments, c, i + 1) for i, c in enumerate(shortlist)]
    prompt = (
        "You are ranking short-form video highlights.\n"
        "Pick the best moments for virality and clarity.\n"
        "Return strict JSON: {\"ranked_indices\":[...],\"reasons\":{\"1\":\"...\"}}\n"
        "Only use provided candidate numbers.\n\n"
        f"Candidates:\n{chr(10).join(lines)}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        r = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        ranked = parsed.get("ranked_indices", [])
        reasons = parsed.get("reasons", {})
    except Exception:
        return candidates[:top_k]

    picked: List[ClipCandidate] = []
    for raw_idx in ranked:
        if not isinstance(raw_idx, int):
            continue
        idx = raw_idx - 1
        if idx < 0 or idx >= len(shortlist):
            continue
        cand = shortlist[idx]
        if str(raw_idx) in reasons and isinstance(reasons[str(raw_idx)], str):
            cand = ClipCandidate(
                start=cand.start,
                end=cand.end,
                score=cand.score,
                title=cand.title,
                reason=f"llm-rerank: {reasons[str(raw_idx)][:120]}",
            )
        overlap = any(not (cand.end <= s.start or cand.start >= s.end) for s in picked)
        if not overlap:
            picked.append(cand)
        if len(picked) >= top_k:
            break

    if not picked:
        return candidates[:top_k]
    return picked
