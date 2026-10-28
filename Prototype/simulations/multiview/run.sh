#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="$HERE/.venv/bin/python"
FF="$($PY -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
SOURCE="$ROOT/Data/Training/Training 8/vidoes/IMG_0108.MOV"

mkdir -p "$HERE/base"
"$FF" -hide_banner -loglevel error -y -threads 4 -i "$SOURCE" -an -frames:v 600 -fps_mode passthrough \
  -vf 'zscale=t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,scale=1280:720:flags=area' \
  -c:v libx264 -profile:v main -preset medium -b:v 2990k -maxrate 3500k -bufsize 6000k -threads 4 \
  "$HERE/base/IMG_0108_600f.mp4"

"$PY" "$HERE/generate_views.py" \
  --base-video "$HERE/base/IMG_0108_600f.mp4" \
  --source-width 3840 \
  --registration "$ROOT/Data/Training/Training 2/court_registration.csv" \
  --roster "$ROOT/Data/Training/Training/games_roster.csv" \
  --ground-truth "$ROOT/Data/Training/Training 4/full player annotations/court position/IMG_0108.txt" \
  --frames 600 --out "$HERE/project"

"$PY" "$HERE/run_all.py"
