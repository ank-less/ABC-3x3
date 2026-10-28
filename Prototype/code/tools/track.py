"""Stage 1: detect and track people in a video at 1280x720 with YOLO + ByteTrack.

    .venv/bin/python tools/track.py IMG_0108 IMG_0109 --weights yolo11l.pt --tag coco_l
    .venv/bin/python tools/track.py IMG_0114 --split challenge --weights best.pt --tag ft_l

Training videos (4K) are downscaled to 1280x720 like tools/extract_frames.py, so the detector sees
the Challenge resolution. Output out/tracks/<tag>/<video>.txt, one row per box in 720p pixels:
    frame,track_id,x,y,w,h,conf
plus <video>.json with the number of decoded frames. Frames are 1-based decode indices
(read sequentially, never seek).

--shrink s (< 1) makes a harder validation set from Training videos: each 720p frame is scaled by s
and pasted top-left on a grey 1280x720 canvas, so players are as small as in the Challenge videos
(~0.65); boxes are mapped back to the full 720p frame before writing.
"""
import argparse
import json
import os
import queue
import threading
import time

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_W, OUT_H = 1280, 720
TRACKER_CFG = os.path.join(os.path.dirname(__file__), "botsort_3x3.yaml")  # chosen 28/10: = ByteTrack on val, fewer breaks on Challenge


def video_path(split, video):
    folder = os.path.join(ROOT, "data", split, "videos")
    return next(os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(video + "."))


def shrink_frame(img, s):
    small = cv2.resize(img, (round(OUT_W * s), round(OUT_H * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(img, 114)
    canvas[:small.shape[0], :small.shape[1]] = small
    return canvas


def frames(path, maxsize=64, shrink=1.0):
    """Decode in a thread (4K HEVC decode is the other bottleneck) and yield 720p BGR frames."""
    q = queue.Queue(maxsize)

    def work():
        cap = cv2.VideoCapture(path)
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if img.shape[1] != OUT_W:
                img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            q.put(shrink_frame(img, shrink) if shrink < 1 else img)
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (img := q.get()) is not None:
        yield img


def track_video(model, path, out_txt, imgsz, conf, classes, shrink=1.0, tracker=TRACKER_CFG, augment=False, throttle=1.0):
    n = 0
    t0 = time.time()
    with open(out_txt, "w") as f:
        for n, img in enumerate(frames(path, shrink=shrink), 1):
            t1 = time.time()
            r = model.track(img, persist=True, tracker=tracker, imgsz=imgsz, conf=conf,
                            classes=classes, half=True, augment=augment, verbose=False)[0]
            time.sleep(throttle * (time.time() - t1))  # idle so the GPU stays under ~60% (machine stays usable)
            if r.boxes.id is None:
                continue
            for (x1, y1, x2, y2), tid, c in zip((r.boxes.xyxy / shrink).tolist(), r.boxes.id.int().tolist(), r.boxes.conf.tolist()):
                f.write(f"{n},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{c:.3f}\n")
            if n % 1000 == 0:
                print(f"  {n} frames, {n / (time.time() - t0):.1f} fps", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--split", default="train", choices=["train", "train720", "challenge"])
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "yolo11l_r3.pt"), help="YOLO weights (final: round 3)")
    ap.add_argument("--tag", required=True, help="output folder name under out/tracks")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--shrink", type=float, default=1.0, help="scale frames down (harder val), see above")
    ap.add_argument("--augment", action="store_true", help="test-time augmentation (flip + scales), ~2x slower")
    ap.add_argument("--tracker", default=TRACKER_CFG, help="tracker yaml (bytetrack_3x3 / botsort_3x3 / botsort_reid_3x3)")
    ap.add_argument("--throttle", type=float, default=1.0, help="sleep this many times each frame's compute time")
    args = ap.parse_args()

    from ultralytics import YOLO
    out_dir = os.path.join(ROOT, "out", "tracks", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    for video in args.videos:
        model = YOLO(args.weights)  # fresh model per video so tracker state does not carry over
        classes = [0] if len(model.names) > 1 else None  # COCO person, or our 1-class player model
        print(f"{video}: tracking with {args.weights}", flush=True)
        t0 = time.time()
        n = track_video(model, video_path(args.split, video), os.path.join(out_dir, f"{video}.txt"),
                        args.imgsz, args.conf, classes, args.shrink, args.tracker, args.augment, args.throttle)
        with open(os.path.join(out_dir, f"{video}.json"), "w") as f:
            json.dump({"video": video, "frames": n, "size": [OUT_W, OUT_H], "weights": args.weights,
                       "imgsz": args.imgsz, "conf": args.conf, "shrink": args.shrink, "tracker": os.path.basename(args.tracker), "augment": args.augment}, f)
        print(f"{video}: {n} frames in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
