"""Build a clean, flat dataset under data/ from the raw Google Drive zip parts and data_redownload/.

    .venv/bin/python tools/prepare_data.py                     # raw Google Drive zip parts (our layout)
    .venv/bin/python tools/prepare_data.py --training "/path/to/Training" --challenge "/path/to/Final challenge"

--training is the official Training folder (vidoes/*.MOV, full player annotations/, court_registration.csv,
games_roster.csv); it may be given several times (the Drive download comes in overlapping zip parts).
--challenge is the folder with IMG_0114.MP4, IMG_0115.MP4 and their csv files.
Raw folders are left untouched. Output:

    data/train/videos/IMG_xxxx.MOV         symlinks to the raw videos
    data/train/bboxes/IMG_xxxx.txt         MOT boxes: frame,id,x,y,w,h,conf,-1,-1 (cleaned)
    data/train/court/IMG_xxxx.txt          GT format: frame,court_x,court_y,team,jersey
    data/train/court_registration.csv, data/train/games_roster.csv
    data/challenge/videos/IMG_xxxx.MP4     symlinks
    data/challenge/court_registration.csv, data/challenge/games_roster.csv

Cleaning (Training only; documented for the method report):
  * boxes whose track id is not in the roster are dropped (IMG_0110 track 0, a spectator);
  * a duplicated (frame, track) keeps the box whose foot point is closest to that track's
    previous frame (IMG_0105, 25 cases);
  * court positions are recomputed from the cleaned boxes with the registration homography.
    This reproduces the official court position files exactly and fills in videos that lack one.
"""
import argparse
import csv
import glob
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from visualize import load_roster  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def homography(path):
    rows = list(csv.DictReader(open(path)))
    img = np.array([[float(r["image_x"]), float(r["image_y"])] for r in rows], np.float64)
    court = np.array([[float(r["court_x"]), float(r["court_y"])] for r in rows], np.float64)
    return cv2.findHomography(img, court)[0]


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def unique_by_name(paths):
    """The zip parts overlap (re-downloads), so keep one file per basename and check duplicates match."""
    out = {}
    for p in sorted(paths):
        name = os.path.basename(p)
        if name in out:
            if os.path.getsize(out[name]) != os.path.getsize(p):
                sys.exit(f"conflicting copies of {name}: {out[name]} vs {p}")
            continue
        out[name] = p
    return out


def clean_boxes(path, players):
    """Return sorted rows [frame, id, x, y, w, h, conf] and a cleaning report."""
    by_key = {}
    dropped = dups = 0
    for line in open(path):
        p = line.strip().split(",")
        frame, tid = int(p[0]), int(p[1])
        if tid not in players:
            dropped += 1
            continue
        by_key.setdefault((frame, tid), []).append([frame, tid] + [float(v) for v in p[2:7]])
    rows = []
    last_foot = {}
    for frame, tid in sorted(by_key):
        cands = by_key[(frame, tid)]
        if len(cands) > 1:
            dups += len(cands) - 1
            prev = last_foot.get(tid)
            if prev is not None:
                cands.sort(key=lambda r: np.hypot(r[2] + r[4] / 2 - prev[0], r[3] + r[5] - prev[1]))
        best = cands[0]
        last_foot[tid] = (best[2] + best[4] / 2, best[3] + best[5])
        rows.append(best)
    return rows, dropped, dups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--training", action="append",
                    help="official Training folder (repeatable); default training/*/Training + data_redownload/")
    ap.add_argument("--challenge", help="official Challenge folder; default 'Final challenge-*/Final challenge'")
    args = ap.parse_args()
    train_dir = os.path.join(ROOT, "data", "train")
    chal_dir = os.path.join(ROOT, "data", "challenge")
    for d in ("videos", "bboxes", "court"):
        os.makedirs(os.path.join(train_dir, d), exist_ok=True)
    os.makedirs(os.path.join(chal_dir, "videos"), exist_ok=True)

    raws = [glob.escape(t) for t in args.training] if args.training else [os.path.join(ROOT, "training", "*", "Training")]
    # data_redownload/ holds files fetched individually (IMG_0107.MOV, extracted annotation zips).
    redl = os.path.join(ROOT, "data_redownload")
    g = lambda *parts: sum((glob.glob(os.path.join(raw, *parts)) for raw in raws), [])
    videos = unique_by_name(g("vidoes", "*.MOV") + g("videos", "*.MOV") + glob.glob(os.path.join(redl, "*.MOV")))
    bboxes = unique_by_name(
        g("full player annotations", "Image-space bounding boxes", "*.txt")
        + glob.glob(os.path.join(redl, "extracted", "Image-space bounding boxes", "*.txt")))
    courts = unique_by_name(
        g("full player annotations", "court position", "*.txt")
        + glob.glob(os.path.join(redl, "extracted", "court position", "*.txt")))
    reg = g("court_registration.csv")[0]
    roster = g("games_roster.csv")[0]
    shutil.copy(reg, os.path.join(train_dir, "court_registration.csv"))
    shutil.copy(roster, os.path.join(train_dir, "games_roster.csv"))
    H = homography(reg)

    for name, src in videos.items():
        link(src, os.path.join(train_dir, "videos", name))

    for name, src in sorted(bboxes.items()):
        video = name[:-4]
        _, players = load_roster(roster, video)
        rows, dropped, dups = clean_boxes(src, players)
        with open(os.path.join(train_dir, "bboxes", name), "w") as f:
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.2f},-1,-1\n")
        feet = np.array([[[r[2] + r[4] / 2, r[3] + r[5]] for r in rows]], np.float64)
        pos = cv2.perspectiveTransform(feet, H)[0]
        gt = sorted((r[0], players[r[1]][0], int(players[r[1]][1]), p) for r, p in zip(rows, pos))
        with open(os.path.join(train_dir, "court", name), "w") as f:
            for frame, team, jersey, (x, y) in gt:
                f.write(f"{frame},{x:.3f},{y:.3f},{team},{jersey}\n")
        check = ""
        if name in courts:
            official = {}
            for line in open(courts[name]):
                p = line.strip().split(",")
                official.setdefault((int(p[0]), p[3], int(p[4])), []).append((float(p[1]), float(p[2])))
            err = max(min(np.hypot(x - a, y - b) for a, b in official[(frame, team, jersey)])
                      for frame, team, jersey, (x, y) in gt)
            check = f", max diff vs official court file {err:.3f} cm"
        else:
            check = ", no official court file (regenerated)"
        print(f"{video}: {len(rows)} boxes, dropped {dropped} off-roster, resolved {dups} duplicates{check}")
        if video + ".MOV" not in videos:
            print(f"  warning: {video}.MOV is missing from the download")

    chal = args.challenge or glob.glob(os.path.join(ROOT, "Final challenge-*", "Final challenge"))[0]
    for src in glob.glob(os.path.join(chal, "*.MP4")):
        link(src, os.path.join(chal_dir, "videos", os.path.basename(src)))
    for f in ("court_registration.csv", "games_roster.csv"):
        shutil.copy(os.path.join(chal, f), os.path.join(chal_dir, f))
    print("challenge videos:", sorted(os.listdir(os.path.join(chal_dir, "videos"))))


if __name__ == "__main__":
    main()
