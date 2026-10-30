#!/usr/bin/env bash
# Score the same pipeline on the held-out Training videos IMG_0108 + IMG_0109 (never trained on), after
# re-encoding them like the Challenge videos (720p SDR H.264, 3 Mbit/s). Needs data/train from
# tools/prepare_data.py. Reported: TI-HOTA 0.9680 (0108 0.9694, 0109 0.9665).
#   ./run_val.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
VIDEOS="IMG_0108 IMG_0109"
for v in $VIDEOS; do [ -f data/train720/videos/$v.mp4 ] || tools/encode720.sh $v; done
$PY tools/track.py $VIDEOS --split train720 --weights weights/yolo11l_r3.pt --tag val_final --throttle "${THROTTLE:-0}"
$PY tools/pipeline.py $VIDEOS --split train720 --tag val_final --model weights/idcls_enc.pt --reclassify
$PY tools/eval_tihota.py --pred out/pred/val_final --videos $VIDEOS
$PY tools/diagnose.py --pred out/pred/val_final --videos $VIDEOS
