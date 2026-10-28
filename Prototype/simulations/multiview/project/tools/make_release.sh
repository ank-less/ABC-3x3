#!/usr/bin/env bash
# Assemble the code + weights ZIP for the paper/code submission (must be the exact version that produced
# the submitted predictions). Run after ./run_challenge.sh:
#   tools/make_release.sh            # -> dist/trackid3x3_release/ and dist/trackid3x3_release.zip
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=dist/trackid3x3_release
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT"/{tools,weights,predictions,data/challenge/videos}
cp README.md requirements.txt run_challenge.sh run_val.sh "$OUT/"
for f in track.py pipeline.py idcls.py eval_tihota.py diagnose.py check_submission.py health.py \
         prepare_data.py extract_frames.py encode720.sh colab_make_enc.py train_yolo.py visualize.py \
         botsort_3x3.yaml make_release.sh; do
  cp tools/$f "$OUT/tools/"
done
cp -L weights/yolo11l_r3.pt weights/idcls_enc.pt "$OUT/weights/"
cp predictions/IMG_0114.txt predictions/IMG_0115.txt "$OUT/predictions/"
cp data/challenge/games_roster.csv data/challenge/court_registration.csv "$OUT/data/challenge/"
echo "Put IMG_0114.MP4 and IMG_0115.MP4 (official Challenge videos) here." > "$OUT/data/challenge/videos/README.txt"
(cd "$OUT/weights" && sha256sum yolo11l_r3.pt idcls_enc.pt > SHA256SUMS)
(cd dist && zip -qr trackid3x3_release.zip trackid3x3_release)
du -sh "$OUT" "$OUT.zip"
