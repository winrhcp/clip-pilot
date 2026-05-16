from __future__ import annotations

import subprocess
from pathlib import Path


def cut_clip(input_video: Path, output_clip: Path, start: float, end: float) -> None:
    duration = max(end - start, 1.0)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.2f}",
        "-i", str(input_video),
        "-t", f"{duration:.2f}",
        "-c:v", "libx264",
        "-c:a", "aac",
        str(output_clip),
    ]
    subprocess.run(cmd, check=True)


def render_vertical_with_subs(input_clip: Path, ass_file: Path, output_clip: Path) -> None:
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        f"subtitles='{ass_file.as_posix()}'"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_clip),
        "-vf", vf,
        "-c:v", "libx264",
        "-c:a", "aac",
        str(output_clip),
    ]
    subprocess.run(cmd, check=True)


def replace_audio_track(input_video: Path, input_audio: Path, output_video: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(input_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)
