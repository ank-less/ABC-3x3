"""Extract downscaled Training frames + YOLO labels for detector fine-tuning on Colab.

    .venv/bin/python tools/extract_frames.py                 # -> data/yolo720/ and data/yolo720.zip

Decoding 4K HEVC is ~12x faster on this machine than on Colab, so frames are extracted here and
only the small dataset is uploaded. Frames are resized to 1280x720, the Challenge resolution.

Layout (Ultralytics):
    images/{train,val}/IMG_xxxx_000123.jpg
    labels/{train,val}/IMG_xxxx_000123.txt     class 0 = player, normalized xc yc w h
    identities/{train,val}.csv                 image,track_id,team,jersey,x,y,w,h (pixels at 720p)
    dataset.yaml                               path: /content/yolo720 (unzip there on Colab)
    manifest.json

The split is by video. Default val = IMG_0108 + IMG_0109, whose team pairings mirror the two
Challenge videos (0114: {6,9,10} vs {6,9,11}; 0115: {4,10,12} vs {1,3,5}).
"""
import argparse
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from visualize import load_boxes, load_roster  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_W, OUT_H = 1280, 720
MIN_SIDE = 4  # px at 720p; boxes smaller than this after clipping are dropped


def extract_video(job):
    video, split, data_dir, out_dir, stride, quality = job
    boxes = load_boxes(os.path.join(data_dir, "bboxes", f"{video}.txt"))
    _, players = load_roster(os.path.join(data_dir, "games_roster.csv"), video)
    cap = cv2.VideoCapture(os.path.join(data_dir, "videos", f"{video}.MOV"))
    src_w, src_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    kx, ky = OUT_W / src_w, OUT_H / src_h
    ident_rows = []
    n_images = n_boxes = 0
    frame = 0
    # Read sequentially: seeking is off by up to 3 frames on some of these videos.
    while cap.grab():
        frame += 1
        if (frame - 1) % stride or frame not in boxes:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        name = f"{video}_{frame:06d}"
        img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        lines = []
        for tid, (x, y, w, h, _) in sorted(boxes[frame].items()):
            x1, y1 = max(0.0, x * kx), max(0.0, y * ky)
            x2, y2 = min(OUT_W, (x + w) * kx), min(OUT_H, (y + h) * ky)
            if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
                continue
            lines.append(f"0 {(x1 + x2) / 2 / OUT_W:.6f} {(y1 + y2) / 2 / OUT_H:.6f} "
                         f"{(x2 - x1) / OUT_W:.6f} {(y2 - y1) / OUT_H:.6f}")
            team, jersey, _ = players[tid]
            ident_rows.append(f"{name}.jpg,{tid},{team},{jersey},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f}")
        with open(os.path.join(out_dir, "labels", split, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_images += 1
        n_boxes += len(lines)
    return video, split, frame, n_images, n_boxes, ident_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "yolo720"))
    ap.add_argument("--stride", type=int, default=5, help="keep every Nth frame (5 = 6 images per second)")
    ap.add_argument("--val", nargs="*", default=["IMG_0108", "IMG_0109"])
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    videos = sorted(os.path.basename(p)[:-4] for p in os.listdir(os.path.join(args.data, "videos")) if p.endswith(".MOV"))
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.out, sub, split), exist_ok=True)
    os.makedirs(os.path.join(args.out, "identities"), exist_ok=True)
    jobs = [(v, "val" if v in args.val else "train", args.data, args.out, args.stride, args.quality) for v in videos]

    idents = {"train": [], "val": []}
    stats = []
    with ProcessPoolExecutor(args.workers) as pool:
        for video, split, frames, n_images, n_boxes, rows in pool.map(extract_video, jobs):
            print(f"{video} [{split}]: {frames} frames decoded, {n_images} images, {n_boxes} boxes", flush=True)
            idents[split] += rows
            stats.append({"video": video, "split": split, "frames": frames, "images": n_images, "boxes": n_boxes})
    for split, rows in idents.items():
        with open(os.path.join(args.out, "identities", f"{split}.csv"), "w") as f:
            f.write("image,track_id,team,jersey,x,y,w,h\n" + "\n".join(rows) + "\n")
    with open(os.path.join(args.out, "dataset.yaml"), "w") as f:
        f.write("# Unzip to /content/yolo720 on Colab, or change path.\n"
                "path: /content/yolo720\ntrain: images/train\nval: images/val\nnames:\n  0: player\n")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"size": [OUT_W, OUT_H], "stride": args.stride, "jpeg_quality": args.quality,
                   "val_videos": args.val, "videos": stats,
                   "source": "official Training split only (cleaned by tools/prepare_data.py)"}, f, indent=2)

    if not args.no_zip:
        zip_path = args.out.rstrip("/") + ".zip"
        base = os.path.dirname(os.path.abspath(args.out))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # JPEGs are already compressed
            for dirpath, _, files in os.walk(args.out):
                for fn in files:
                    full = os.path.join(dirpath, fn)
                    z.write(full, os.path.relpath(full, base))
        print(f"wrote {zip_path} ({os.path.getsize(zip_path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
