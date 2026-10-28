#!/usr/bin/env bash
# Re-encode Training videos the way the Challenge videos look: 4K HDR (HLG, 10 bit) -> 1280x720 SDR
# (bt709, hable tone mapping) H.264 Main at ~3 Mbit/s (Challenge: h264 Main, 2990 kb/s).
#   tools/light.sh tools/encode720.sh IMG_0108 IMG_0109       # -> data/train720/videos/IMG_xxxx.mp4
set -e
cd "$(dirname "$0")/.."
FF=$(.venv/bin/python -c "import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())")
mkdir -p data/train720/videos
# split "train720": same roster and GT as data/train, registration in 720p pixels (4K / 3)
ln -sfn ../train/games_roster.csv data/train720/games_roster.csv
ln -sfn ../train/court data/train720/court
.venv/bin/python -c "
import csv; r = list(csv.DictReader(open('data/train/court_registration.csv')))
print('point_id,image_x,image_y,court_x,court_y')
for p in r: print(f\"{p['point_id']},{float(p['image_x']) / 3:.3f},{float(p['image_y']) / 3:.3f},{float(p['court_x'])},{float(p['court_y'])}\")
" > data/train720/court_registration.csv
for v in "$@"; do
  src=$(realpath data/train/videos/$v.*)
  "$FF" -hide_banner -loglevel error -y -threads 4 -i "$src" -an -fps_mode passthrough \
    -vf "zscale=t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,scale=1280:720:flags=area" \
    -c:v libx264 -profile:v main -preset medium -b:v 2990k -maxrate 3500k -bufsize 6000k -threads 4 \
    data/train720/videos/$v.mp4
  echo "$v done"
done
