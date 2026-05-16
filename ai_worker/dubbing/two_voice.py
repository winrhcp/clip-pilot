from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path

import numpy as np

from ai_worker.dubbing.tts import synthesize_thai_voice
from ai_worker.whisper.transcriber import Segment


def _extract_pcm_wav(input_video: Path, output_wav: Path, sample_rate: int = 16000) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _read_wav_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        data = wf.readframes(n)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def _estimate_pitch_hz(segment_samples: np.ndarray, sr: int) -> float | None:
    if segment_samples.size < int(sr * 0.2):
        return None
    x = segment_samples - float(np.mean(segment_samples))
    energy = float(np.sqrt(np.mean(x * x)))
    if energy < 0.01:
        return None

    min_hz = 80.0
    max_hz = 300.0
    min_lag = max(1, int(sr / max_hz))
    max_lag = max(min_lag + 1, int(sr / min_hz))
    if x.size <= max_lag:
        return None

    corr = np.correlate(x, x, mode="full")[x.size - 1 :]
    search = corr[min_lag:max_lag]
    if search.size == 0:
        return None
    lag = int(np.argmax(search)) + min_lag
    peak = float(corr[lag])
    if peak <= 0:
        return None
    return float(sr / lag)


def _gender_from_pitch(pitch_hz: float | None) -> str:
    if pitch_hz is None:
        return "unknown"
    return "female" if pitch_hz >= 165.0 else "male"


def _ffprobe_duration_seconds(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float((r.stdout or "0").strip() or 0.0)


def _atempo_filter(speed: float) -> str:
    speed = max(speed, 0.1)
    parts: list[str] = []
    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    parts.append(f"atempo={speed:.4f}")
    return ",".join(parts)


def _create_silence_wav(out_path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    dur = max(duration_sec, 0.01)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl=mono",
        "-t",
        f"{dur:.3f}",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _retime_speech_to_duration(input_audio: Path, output_audio: Path, target_duration: float) -> None:
    actual = max(_ffprobe_duration_seconds(input_audio), 0.01)
    target = max(target_duration, 0.05)
    speed = actual / target
    filter_chain = _atempo_filter(speed)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-af",
        filter_chain,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_audio),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_two_voice_thai_dub_track(input_clip: Path, thai_segments: list[Segment], output_audio: Path) -> None:
    male_voice = os.getenv("TTS_MALE_VOICE", "onyx")
    female_voice = os.getenv("TTS_FEMALE_VOICE", "shimmer")

    work_dir = output_audio.parent / f"{output_audio.stem}_parts"
    work_dir.mkdir(parents=True, exist_ok=True)

    wav_for_analysis = work_dir / "source_mono.wav"
    _extract_pcm_wav(input_clip, wav_for_analysis, sample_rate=16000)
    mono, sr = _read_wav_mono(wav_for_analysis)

    parts: list[Path] = []
    cursor = 0.0
    for idx, seg in enumerate(thai_segments, 1):
        start = max(seg.start, 0.0)
        end = max(seg.end, start + 0.05)
        if start > cursor:
            silence = work_dir / f"{idx:03d}_silence.wav"
            _create_silence_wav(silence, start - cursor, sample_rate=16000)
            parts.append(silence)

        s0 = min(int(start * sr), mono.size)
        s1 = min(int(end * sr), mono.size)
        pitch = _estimate_pitch_hz(mono[s0:s1], sr) if s1 > s0 else None
        gender = _gender_from_pitch(pitch)
        voice = female_voice if gender == "female" else male_voice

        tts_mp3 = work_dir / f"{idx:03d}_tts.mp3"
        tts_wav = work_dir / f"{idx:03d}_speech.wav"
        synthesize_thai_voice(seg.text, tts_mp3, voice=voice)
        _retime_speech_to_duration(tts_mp3, tts_wav, end - start)
        parts.append(tts_wav)
        cursor = end

    if not parts:
        raise RuntimeError("No segments available for Thai dubbing.")

    concat_list = work_dir / "concat.txt"
    concat_lines = [f"file '{p.resolve().as_posix()}'" for p in parts]
    concat_list.write_text("\n".join(concat_lines), encoding="utf-8")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c:a",
        "mp3",
        str(output_audio),
    ]
    subprocess.run(cmd, check=True)
