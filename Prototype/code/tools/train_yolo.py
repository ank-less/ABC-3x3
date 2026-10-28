"""Fine-tune the player detector (YOLO11l, 1 class) in three rounds, as run on Colab (NVIDIA L4).

    python tools/train_yolo.py half --data /content/yolo720                 # train_half.yaml (round 1 data)
    python tools/train_yolo.py r1 --data /content/yolo720 --project runs    # COCO yolo11l.pt -> r1
    python tools/train_yolo.py r2 --data /content/yolo720 --project runs    # r1 best.pt -> r2
    python tools/train_yolo.py r3 --data /content/yolo720 --project runs    # r2 last.pt -> r3 (final)

Data (Training split only, val = IMG_0108 + IMG_0109):
  * data/yolo720 from tools/extract_frames.py: every 5th frame of the 4K videos downscaled to 1280x720;
  * round 3 uses images/{train,val}_enc from tools/colab_make_enc.py: the same frames and labels taken
    from the Challenge-style re-encodes (tools/encode720.sh: 720p SDR H.264, 3 Mbit/s).
The arguments below are the non-default ones stored in the released checkpoints (train_args).
Round 1 stopped early at epoch 10 (patience 8, best = epoch 2); round 2 starts from that best.pt.
The final detector is <project>/ft_yolo11l_1280_r3_enc/weights/last.pt (= weights/yolo11l_r3.pt).
"""
import argparse
import os

ROUNDS = {
    "r1": dict(model="yolo11l.pt", data="train_half.yaml", name="ft_yolo11l_1280",
               epochs=30, patience=8, batch=5, imgsz=1280, optimizer="AdamW", lr0=0.002, cos_lr=True,
               warmup_bias_lr=0.0, close_mosaic=5, scale=0.6),
    "r2": dict(model="{project}/ft_yolo11l_1280/weights/best.pt", data="dataset.yaml", name="ft_yolo11l_1280_r2",
               epochs=8, patience=0, batch=5, imgsz=1280, optimizer="SGD", lr0=0.002, lrf=0.1, warmup_epochs=0,
               close_mosaic=3, hsv_h=0.0, scale=0.3, mosaic=0.5, save_period=1),
    "r3": dict(model="{project}/ft_yolo11l_1280_r2/weights/last.pt", data="dataset_enc.yaml",
               name="ft_yolo11l_1280_r3_enc", epochs=6, patience=0, batch=5, imgsz=1280, optimizer="SGD",
               lr0=0.002, lrf=0.1, close_mosaic=2, hsv_h=0.0, degrees=5, shear=3, perspective=0.0005, scale=0.4,
               mosaic=0.5, save_period=1),
}


def make_half(data):
    """Consecutive sampled frames are near duplicates: keep every 2nd train image of each video."""
    names = sorted(os.listdir(os.path.join(data, "images", "train")))
    by_video = {}
    for n in names:
        by_video.setdefault(n[:8], []).append(n)
    keep = [os.path.join(data, "images", "train", n) for v in sorted(by_video) for n in by_video[v][::2]]
    with open(os.path.join(data, "train_half.txt"), "w") as f:
        f.write("\n".join(keep) + "\n")
    with open(os.path.join(data, "train_half.yaml"), "w") as f:
        f.write(f"path: {data}\ntrain: train_half.txt\nval: images/val\nnames:\n  0: player\n")
    print(f"train_half: {len(keep)} of {len(names)} images")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("round", choices=["half", *ROUNDS])
    ap.add_argument("--data", default="/content/yolo720", help="unzipped data/yolo720 (with dataset*.yaml)")
    ap.add_argument("--project", default="runs", help="output folder of the Ultralytics runs")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.round == "half":
        return make_half(args.data)
    from ultralytics import YOLO
    cfg = dict(ROUNDS[args.round])
    model = cfg.pop("model").format(project=args.project)
    cfg["data"] = os.path.join(args.data, cfg["data"])
    YOLO(model).train(**cfg, project=args.project, exist_ok=True, workers=args.workers, seed=0)


if __name__ == "__main__":
    main()
