param()
python -m pip install -r ai_worker/requirements.txt
python -m ai_worker.main --input videos/input/sample.mp4 --output-dir videos/output --model medium --top-k 3
