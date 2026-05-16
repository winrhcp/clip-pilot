# ClipPilot

Local MVP for AI highlight clipping based on your blueprint.

## What it does
- Transcribes long-form video with `faster-whisper`
- Scores likely highlight moments (heuristic or LLM hybrid rerank)
- Cuts top clips with `ffmpeg`
- Generates subtitles (`.ass`)
- Optional Thai subtitle translation (`--subtitle-lang th`)
- Optional Thai dubbing (`--dub-lang th --dub-mode replace`)
  - For interview-style clips, dubbing uses two voices (male/female) by estimating speaker pitch per segment.
- Renders 9:16 output clips with burned subtitles
- Exports `timeline.json`

## Structure
- `ai_worker/main.py`: one-command pipeline
- `ai_worker/whisper/transcriber.py`: speech-to-text
- `ai_worker/scoring/highlight_scoring.py`: highlight ranking
- `ai_worker/subtitles/ass_writer.py`: subtitle generation
- `ai_worker/reframing/ffmpeg_render.py`: cutting + vertical render

## Setup (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ai_worker\requirements.txt
```

Install ffmpeg and ensure `ffmpeg` is on `PATH`.

## วิธีใช้งานตอนนี้ (Windows PowerShell)
1. เปิด PowerShell ที่โฟลเดอร์โปรเจกต์ `d:\projects-win\clip-pilot`
2. สร้าง virtual environment และติดตั้ง dependency
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ai_worker\requirements.txt
```
3. เช็กว่าเครื่องมองเห็น ffmpeg
```powershell
ffmpeg -version
```
4. วางไฟล์วิดีโอที่ต้องการตัดไฮไลต์ใน `videos/input/`
5. รัน pipeline
```powershell
python -m ai_worker.main --input videos/input/your-video.mp4 --output-dir videos/output --model medium --top-k 3 --scoring hybrid
```

Thai subtitle output:

```powershell
python -m ai_worker.main --input videos/input/your-video.mp4 --output-dir videos/output --model medium --top-k 3 --scoring hybrid --subtitle-lang th
```

Thai dubbing (replace source audio):

```powershell
python -m ai_worker.main --input videos/input/your-video.mp4 --output-dir videos/output --model medium --top-k 3 --scoring hybrid --subtitle-lang th --dub-lang th --dub-mode replace
```
python -m ai_worker.main --input videos/input/Sam_Altman_Shows_Me_GPT_5_And_Whats_Next.mp4 --output-dir videos/output --model medium --top-k 5 --scoring hybrid --subtitle-lang th --dub-lang th --dub-mode replace
## Run
Put your source file in `videos/input/`, then:

```powershell
python -m ai_worker.main --input videos/input/your-video.mp4 --output-dir videos/output --model medium --top-k 3 --scoring hybrid
```

Scoring mode:
- `--scoring heuristic`: ใช้คีย์เวิร์ด + punctuation เท่านั้น
- `--scoring hybrid` (default): heuristic shortlist แล้วให้ LLM rerank
- `--scoring llm`: เหมือน hybrid (ถ้าไม่มี LLM config จะ fallback heuristic)

LLM config (optional, required for hybrid/llm rerank):
```powershell
$env:LLM_API_KEY="your_api_key"
$env:LLM_MODEL="gpt-4o-mini"
$env:LLM_BASE_URL="https://api.openai.com/v1"
```

Thai subtitle translation requires `LLM_API_KEY` as well. If not set, it falls back to source-language subtitles.

Thai dubbing also requires `LLM_API_KEY`. Optional TTS env:
```powershell
$env:TTS_MODEL="gpt-4o-mini-tts"
$env:TTS_VOICE="alloy"
$env:TTS_MALE_VOICE="onyx"
$env:TTS_FEMALE_VOICE="shimmer"
```

## Output
- `videos/output/clip_01.mp4` ...
- `videos/output/timeline.json`
- จะได้หลายคลิปตามจำนวน `--top-k`

## Notes
- This is V1 MVP focused on pipeline quality.
- Next upgrades: scene detection gating, face tracking crop, Go backend API + queue.
