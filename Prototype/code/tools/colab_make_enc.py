"""On Colab: build the Challenge-style YOLO split from the re-encoded Training videos.

    python colab_make_enc.py --videos /content/drive/MyDrive/UWB-work/train720 --data /content/yolo720

For every labelled image of data/yolo720 (images/{train,val}/IMG_xxxx_000123.jpg, every 5th frame),
take the same frame from the re-encoded video (tools/encode720.sh: 720p SDR H.264 at 3 Mbit/s, like
the Challenge videos) and write images/{train,val}_enc/ + labels/{train,val}_enc/ with the same labels,
plus dataset_enc.yaml (train = train_enc, val = val_enc). Frames are read sequentially (no seeking).
"""
import argparse
import os
import shutil

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder with IMG_xxxx.mp4 from tools/encode720.sh")
    ap.add_argument("--data", default="/content/yolo720")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()
    for split in ("train", "val"):
        lab_dir = os.path.join(args.data, "labels", split)
        wanted = {}
        for name in os.listdir(lab_dir):
            wanted.setdefault(name[:8], set()).add(int(name[9:15]))
        img_out = os.path.join(args.data, "images", f"{split}_enc")
        lab_out = os.path.join(args.data, "labels", f"{split}_enc")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lab_out, exist_ok=True)
        for video, frames in sorted(wanted.items()):
            cap = cv2.VideoCapture(os.path.join(args.videos, f"{video}.mp4"))
            n = written = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                n += 1
                if n in frames:
                    name = f"{video}_{n:06d}"
                    cv2.imwrite(os.path.join(img_out, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                    shutil.copy(os.path.join(lab_dir, name + ".txt"), os.path.join(lab_out, name + ".txt"))
                    written += 1
            print(f"{split} {video}: {n} frames decoded, {written}/{len(frames)} images written", flush=True)
    with open(os.path.join(args.data, "dataset_enc.yaml"), "w") as f:
        f.write(f"path: {args.data}\ntrain: images/train_enc\nval: images/val_enc\nnames:\n  0: player\n")
    print("wrote", os.path.join(args.data, "dataset_enc.yaml"))


if __name__ == "__main__":
    main()
