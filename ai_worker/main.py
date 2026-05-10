from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_worker.reframing.ffmpeg_render import cut_clip, render_vertical_with_subs
from ai_worker.scoring.highlight_scoring import rerank_with_llm, score_highlights, serialize_candidates
from ai_worker.subtitles.ass_writer import write_ass
from ai_worker.subtitles.translator import translate_segments_to_thai
from ai_worker.whisper.transcriber import Segment, transcribe


def _segments_in_range(segments: list[Segment], start: float, end: float) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        if seg.end < start or seg.start > end:
            continue
        out.append(Segment(start=max(seg.start - start, 0.0), end=max(seg.end - start, 0.1), text=seg.text))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Highlight Clipper MVP")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output-dir", default="videos/output", help="Output directory")
    parser.add_argument("--model", default="medium", help="faster-whisper model")
    parser.add_argument("--top-k", type=int, default=3, help="Number of clips")
    parser.add_argument(
        "--subtitle-lang",
        choices=["source", "th"],
        default="source",
        help="Subtitle language: source transcript or Thai translation",
    )
    parser.add_argument(
        "--scoring",
        choices=["heuristic", "hybrid", "llm"],
        default="hybrid",
        help="Highlight scoring mode",
    )
    args = parser.parse_args()

    input_video = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    temp_dir = output_dir / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    segments = transcribe(input_video, model_name=args.model)
    heuristic_candidates = score_highlights(segments, top_k=max(args.top_k * 4, 12))
    if args.scoring == "heuristic":
        candidates = heuristic_candidates[: args.top_k]
    else:
        candidates = rerank_with_llm(segments, heuristic_candidates, top_k=args.top_k)

    outputs = []
    for i, cand in enumerate(candidates, 1):
        raw_clip = temp_dir / f"clip_{i:02d}_raw.mp4"
        ass_file = temp_dir / f"clip_{i:02d}.ass"
        final_clip = output_dir / f"clip_{i:02d}.mp4"

        cut_clip(input_video, raw_clip, cand.start, cand.end)
        clip_segments = _segments_in_range(segments, cand.start, cand.end)
        if args.subtitle_lang == "th":
            clip_segments = translate_segments_to_thai(clip_segments)
        write_ass(clip_segments, ass_file)
        render_vertical_with_subs(raw_clip, ass_file, final_clip)

        outputs.append(
            {
                "index": i,
                "start": cand.start,
                "end": cand.end,
                "score": cand.score,
                "title": cand.title,
                "reason": cand.reason,
                "output": str(final_clip),
            }
        )

    timeline = {"clips": outputs, "candidates": serialize_candidates(candidates)}
    (output_dir / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(timeline, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
