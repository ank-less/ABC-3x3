#!/usr/bin/env bash
# Reproduce the submitted predictions from the two Challenge videos (about 25 min on an RTX 4050):
#   ./run_challenge.sh                 # -> predictions/IMG_0114.txt, predictions/IMG_0115.txt
# Needs data/challenge/{videos/IMG_0114.MP4, videos/IMG_0115.MP4, games_roster.csv, court_registration.csv}
# and weights/{yolo11l_r3.pt, idcls_enc.pt}. THROTTLE=1 halves GPU load (same output, twice as slow).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
VIDEOS="IMG_0114 IMG_0115"
# 1. detect + track people at 1280x720 (YOLO11l round 3 + BoT-SORT, tools/botsort_3x3.yaml)
$PY tools/track.py $VIDEOS --split challenge --weights weights/yolo11l_r3.pt --tag final --throttle "${THROTTLE:-0}"
# 2. identify (bib colour + number), assign the 6 roster identities per frame, project to the court,
#    clean outliers, interpolate -> 6 rows per frame
$PY tools/pipeline.py $VIDEOS --split challenge --tag final --model weights/idcls_enc.pt --reclassify --out predictions
# 3. check the submission format against the videos and rosters
$PY tools/check_submission.py --pred predictions --split challenge
