"""Player identity classifier: bib colour + jersey number from a single box crop.

    .venv/bin/python tools/idcls.py negatives      # person boxes that are not players (Training only)
    .venv/bin/python tools/idcls.py cache          # crops -> data/idcls/{train,val}.npz
    .venv/bin/python tools/idcls.py train          # -> data/idcls/model.pt

Each of the four teams wears a fixed bib colour in every video, so an identity is (colour, number):
    purple {4,10,12}, yellow {6,9,10}, green {6,9,11}, pink {1,3,5}
That gives 12 identity classes plus "other" (spectators, referees, people in non-player bibs).
The roster of a video picks 6 of the 12; colour also tells which roster team (A/B) a box belongs to.
Trained on the Training split only (data/yolo720, val = IMG_0108 + IMG_0109).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "yolo720")
OUT = os.path.join(ROOT, "data", "idcls")

TEAM_SETS = {"purple": (4, 10, 12), "yellow": (6, 9, 10), "green": (6, 9, 11), "pink": (1, 3, 5)}
CLASSES = [(c, n) for c, nums in TEAM_SETS.items() for n in nums]  # 12 identities
OTHER = len(CLASSES)
N_CLASSES = OTHER + 1
CROP_H, CROP_W = 128, 64  # cached crop size
IN_H, IN_W = 160, 80      # network input size
PAD = 0.1                 # box is enlarged by this fraction on each side before cropping


def colour_of(jerseys):
    """Bib colour of a team given its three jersey numbers."""
    key = tuple(sorted(int(j) for j in jerseys))
    return next(c for c, nums in TEAM_SETS.items() if tuple(sorted(nums)) == key)


def roster_classes(teams):
    """{team: [class index, ...]} for a roster {team: [jersey, ...]}."""
    out = {}
    for team, nums in teams.items():
        c = colour_of(nums)
        out[team] = [CLASSES.index((c, int(n))) for n in nums]
    return out


def load_team_rosters(path):
    """{video: {team: [jersey, ...]}} from either games_roster.csv (Training or Challenge)."""
    out = {}
    for r in csv.DictReader(open(path)):
        teams = {}
        for part in r["team: jersey"].split(";"):
            name, nums = part.split(":")
            teams[name.strip()] = [int(n) for n in nums.split(",")]
        out[r["video"]] = teams
    return out


def crop(img, x, y, w, h, size=(CROP_W, CROP_H)):
    """Crop a box enlarged by PAD (zero padded at the image border) and resize to size."""
    x1, y1 = int(round(x - PAD * w)), int(round(y - PAD * h))
    x2, y2 = int(round(x + w + PAD * w)), int(round(y + h + PAD * h))
    H, W = img.shape[:2]
    out = np.zeros((y2 - y1, x2 - x1, 3), np.uint8)
    sx1, sy1, sx2, sy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = img[sy1:sy2, sx1:sx2]
    return cv2.resize(out, size, interpolation=cv2.INTER_AREA)


def build_net(pretrained=True):
    import torch.nn as nn
    import torchvision
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, N_CLASSES)
    return net


def _best_device(device=None):
    if device is not None:
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_net(path, device=None):
    import torch
    device = _best_device(device)
    # The submitted classifier checkpoint contains all parameters, so inference needs no ImageNet download.
    net = build_net(pretrained=False)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(device).eval()


MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def to_tensor(crops):
    """uint8 BGR crops (N, CROP_H, CROP_W, 3) -> normalized float tensor (N, 3, IN_H, IN_W)."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(crops[..., ::-1])).permute(0, 3, 1, 2).float()
    x = torch.nn.functional.interpolate(x, size=(IN_H, IN_W), mode="bilinear", align_corners=False)
    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


def predict(net, crops, device=None, batch=512):
    """Softmax probabilities (N, N_CLASSES) for uint8 BGR crops."""
    import torch
    out = []
    device = _best_device(device or str(next(net.parameters()).device))
    amp = torch.autocast("cuda", dtype=torch.float16) if str(device).startswith("cuda") else torch.no_grad()
    with torch.no_grad(), amp:
        for i in range(0, len(crops), batch):
            out.append(torch.softmax(net(to_tensor(crops[i:i + batch]).to(device)).float(), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_CLASSES), np.float32)


# ---------------------------------------------------------------- data preparation

def iou(a, b):
    """IoU between boxes a (n,4) and b (m,4) in x,y,w,h."""
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0, None], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1, None], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter)


def read_identities(split):
    by_image = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(DATA, "identities", f"{split}.csv"))):
        by_image[r["image"]].append(r)
    return by_image


def cmd_negatives(args):
    """Run a COCO person detector on Training images; boxes far from every player are 'other'."""
    from ultralytics import YOLO
    model = YOLO(args.weights)
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "val"):
        by_image = read_identities(split)
        names = sorted(by_image)[::args.every]
        rows = []
        for i in range(0, len(names), 16):
            batch = names[i:i + 16]
            res = model.predict([os.path.join(DATA, "images", split, n) for n in batch], imgsz=1280,
                                conf=0.3, classes=[0], half=True, verbose=False)
            for name, r in zip(batch, res):
                det = r.boxes.xywh.cpu().numpy().copy()
                if not len(det):
                    continue
                det[:, :2] -= det[:, 2:] / 2
                gt = np.array([[float(g[k]) for k in "xywh"] for g in by_image[name]])
                far = iou(det, gt).max(1) < 0.1
                rows += [(name, *d) for d in det[far] if d[3] >= 20]
        with open(os.path.join(OUT, f"negatives_{split}.csv"), "w") as f:
            f.write("image,x,y,w,h\n" + "".join(f"{n},{x:.1f},{y:.1f},{w:.1f},{h:.1f}\n" for n, x, y, w, h in rows))
        print(f"{split}: {len(rows)} negative boxes from {len(names)} images")


def cmd_cache(args):
    rosters = load_team_rosters(os.path.join(ROOT, "data", "train", "games_roster.csv"))
    for split in ("train", "val"):
        by_image = read_identities(split)
        negs = defaultdict(list)
        for r in csv.DictReader(open(os.path.join(OUT, f"negatives_{split}.csv"))):
            negs[r["image"]].append(r)
        names = sorted(set(by_image) | set(negs))
        crops, labels, videos = [], [], []
        for k, name in enumerate(names):
            img = cv2.imread(os.path.join(DATA, "images", split, name))
            video = name[:8]
            rcls = roster_classes(rosters[video])
            for r in by_image.get(name, []):
                crops.append(crop(img, *(float(r[c]) for c in "xywh")))
                labels.append(rcls[r["team"]][rosters[video][r["team"]].index(int(r["jersey"]))])
                videos.append(video)
            for r in negs.get(name, []):
                crops.append(crop(img, *(float(r[c]) for c in "xywh")))
                labels.append(OTHER)
                videos.append(video)
            if k % 2000 == 0:
                print(f"  {split} {k}/{len(names)} images", flush=True)
        np.savez(os.path.join(OUT, f"{split}.npz"), crops=np.stack(crops), labels=np.array(labels, np.int64),
                 videos=np.array(videos))
        print(f"{split}: {len(crops)} crops, class counts {np.bincount(labels, minlength=N_CLASSES).tolist()}")


def cmd_cache_video(args):
    """Same boxes and labels as `cache`, but cropped from the Challenge-style re-encoded videos
    (tools/encode720.sh: 720p SDR H.264 at 3 Mbit/s) -> data/idcls/{train,val}_enc.npz."""
    rosters = load_team_rosters(os.path.join(ROOT, "data", "train", "games_roster.csv"))
    for split in ("train", "val"):
        by_image = read_identities(split)
        negs = defaultdict(list)
        for r in csv.DictReader(open(os.path.join(OUT, f"negatives_{split}.csv"))):
            negs[r["image"]].append(r)
        by_video = defaultdict(dict)  # video -> {frame: image name}
        for name in set(by_image) | set(negs):
            by_video[name[:8]][int(name[9:15])] = name
        crops, labels, videos = [], [], []
        for video in sorted(by_video):
            rcls = roster_classes(rosters[video])
            cap = cv2.VideoCapture(os.path.join(ROOT, "data", "train720", "videos", f"{video}.mp4"))
            frame = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                frame += 1
                name = by_video[video].get(frame)
                if name is None:
                    continue
                for r in by_image.get(name, []):
                    crops.append(crop(img, *(float(r[c]) for c in "xywh")))
                    labels.append(rcls[r["team"]][rosters[video][r["team"]].index(int(r["jersey"]))])
                    videos.append(video)
                for r in negs.get(name, []):
                    crops.append(crop(img, *(float(r[c]) for c in "xywh")))
                    labels.append(OTHER)
                    videos.append(video)
            print(f"  {split} {video}: {frame} frames, {len(crops)} crops so far", flush=True)
        np.savez(os.path.join(OUT, f"{split}_enc.npz"), crops=np.stack(crops), labels=np.array(labels, np.int64),
                 videos=np.array(videos))
        print(f"{split}_enc: {len(crops)} crops")


# ---------------------------------------------------------------- training

def augment(batch, rng, strong=False):
    """Simulate the Challenge view: smaller players (down/up-scaling), box jitter, light changes.
    No horizontal flip (mirrored digits) and almost no hue shift (hue is the team cue).
    strong: also the Challenge camera (other viewpoint: slight rotation) and its SDR/H.264 look
    (darker, less saturated, gamma, compression blocks)."""
    out = np.empty_like(batch)
    for i, c in enumerate(batch):
        h, w = c.shape[:2]
        # box jitter: random shift/scale of the crop window
        s = rng.uniform(0.9, 1.1)
        dx, dy = rng.uniform(-0.08, 0.08) * w, rng.uniform(-0.06, 0.06) * h
        M = np.array([[s, 0, (1 - s) * w / 2 + dx], [0, s, (1 - s) * h / 2 + dy]], np.float32)
        if strong:
            M = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-6, 6), 1.0), [0, 0, 1]])[:2] @ np.vstack([M, [0, 0, 1]])
            M = M.astype(np.float32)
        c = cv2.warpAffine(c, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        # resolution: Challenge players are ~50-80 px tall vs ~70-120 px in Training at 720p
        th = int(rng.uniform(36, h))
        c = cv2.resize(cv2.resize(c, (max(8, th * w // h), th), interpolation=cv2.INTER_AREA), (w, h))
        if rng.random() < 0.3:
            c = cv2.GaussianBlur(c, (3, 3), 0)
        # brightness / contrast / saturation
        lo_b, lo_s = (0.55, 0.45) if strong else (0.7, 0.75)
        f = c.astype(np.float32) * rng.uniform(lo_b, 1.3) + rng.uniform(-25, 25)
        gray = f.mean(2, keepdims=True)
        f = gray + (f - gray) * rng.uniform(lo_s, 1.25)
        f = np.clip(f, 0, 255)
        if strong:
            f = 255 * (f / 255) ** rng.uniform(0.75, 1.4)
        c = f.astype(np.uint8)
        if strong and rng.random() < 0.3:
            c = cv2.imdecode(cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(30, 75))])[1], cv2.IMREAD_COLOR)
        out[i] = c
    return out


def evaluate(net, crops, labels, videos, rosters):
    probs = predict(net, crops)
    pred = probs.argmax(1)
    res = {"acc13": float((pred == labels).mean())}
    colour = lambda k: np.array([CLASSES[i][0] if i < OTHER else "other" for i in k])
    res["colour_acc"] = float((colour(pred) == colour(labels)).mean())
    # restricted to the video's 6 roster identities + other, as the pipeline uses it
    ok = []
    for v in np.unique(videos):
        m = videos == v
        allowed = sum(roster_classes(rosters[v]).values(), []) + [OTHER]
        sub = probs[m][:, allowed]
        ok.append(np.array(allowed)[sub.argmax(1)] == labels[m])
    res["roster_acc"] = float(np.concatenate(ok).mean())
    return res, probs


def cmd_train(args):
    import torch
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    import time
    tr, va = np.load(os.path.join(OUT, "train.npz")), np.load(os.path.join(OUT, "val.npz"))
    xtr, ytr = tr["crops"], tr["labels"]
    xva, yva, vva = va["crops"], va["labels"], va["videos"]
    clean_val = None
    if args.enc:  # add the Challenge-style crops; select on the Challenge-style val, also report clean val
        te, ve = np.load(os.path.join(OUT, "train_enc.npz")), np.load(os.path.join(OUT, "val_enc.npz"))
        xtr, ytr = np.concatenate([xtr, te["crops"]]), np.concatenate([ytr, te["labels"]])
        clean_val = (xva, yva, vva)
        xva, yva, vva = ve["crops"], ve["labels"], ve["videos"]
    rosters = load_team_rosters(os.path.join(ROOT, "data", "train", "games_roster.csv"))
    # balance: every identity class equally likely, 'other' at the same rate as one identity
    counts = np.bincount(ytr, minlength=N_CLASSES)
    weights = (1.0 / counts)[ytr]
    weights /= weights.sum()
    net = build_net().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    best = -1
    for ep in range(args.epochs):
        net.train()
        tot = 0.0
        for _ in range(args.steps):
            idx = rng.choice(len(ytr), args.batch, p=weights)
            t0 = time.time()
            x = to_tensor(augment(xtr[idx], rng, args.strong)).cuda(non_blocking=True)
            y = torch.from_numpy(ytr[idx]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(net(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            if args.throttle:  # leave the GPU/CPU idle part of the time (keep the machine usable)
                time.sleep(args.throttle * (time.time() - t0))
        net.eval()
        res, probs = evaluate(net, xva, yva, vva, rosters)
        pos = yva != OTHER
        pred = probs.argmax(1)
        print(f"epoch {ep + 1}: loss {tot / args.steps:.3f}  val acc13 {res['acc13']:.3f}  colour {res['colour_acc']:.3f}  "
              f"roster-restricted {res['roster_acc']:.3f}  players-only acc {float((pred[pos] == yva[pos]).mean()):.3f}  "
              f"other recall {float((pred[~pos] == OTHER).mean()):.3f}", flush=True)
        if clean_val is not None:
            print(f"    clean val roster-restricted {evaluate(net, *clean_val, rosters)[0]['roster_acc']:.3f}", flush=True)
        if res["roster_acc"] > best:
            best = res["roster_acc"]
            torch.save(net.state_dict(), os.path.join(OUT, args.out))
    print(f"best roster-restricted val acc {best:.3f} -> {os.path.join(OUT, args.out)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("negatives")
    p.add_argument("--weights", default="yolo11l.pt")
    p.add_argument("--every", type=int, default=3, help="use every Nth image")
    sub.add_parser("cache")
    sub.add_parser("cache-video")
    p = sub.add_parser("train")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--steps", type=int, default=400, help="batches per epoch")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--enc", action="store_true", help="also train on {train,val}_enc.npz (Challenge-style video)")
    p.add_argument("--strong", action="store_true", help="stronger colour/rotation/compression augmentation")
    p.add_argument("--throttle", type=float, default=0.0, help="sleep this many times each step's duration")
    p.add_argument("--out", default="model.pt", help="file name in data/idcls/")
    args = ap.parse_args()
    {"negatives": cmd_negatives, "cache": cmd_cache, "cache-video": cmd_cache_video, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
