"""Stages 2-3: identify tracked boxes and write a submission file per video.

    .venv/bin/python tools/pipeline.py IMG_0108 IMG_0109 --tag coco_l          # Training val videos
    .venv/bin/python tools/pipeline.py IMG_0114 IMG_0115 --split challenge --tag ft_l
    .venv/bin/python tools/eval_tihota.py --pred out/pred/coco_l --videos IMG_0108 IMG_0109

Input: out/tracks/<tag>/<video>.txt from tools/track.py (720p boxes).
  1. classify: every box crop -> probabilities over the 12 (colour, number) identities + other
     (tools/idcls.py), cached as out/tracks/<tag>/<video>.probs.npy;
  2. restrict to the video's roster: the 6 roster identities, everything else counts as "other";
  3. box -> identity: smooth the probabilities along each tracklet (31-frame moving average), then
     per frame Hungarian-match boxes to the 6 identities (one box per identity per frame);
  4. court position = homography(bottom-centre of the box); positions more than 60 cm off the running
     median of the identity's own track (15 observations) are dropped as identity spikes, then short
     stretches reached only by impossible jumps (wrong player, or feet in the air) are dropped; gaps are
     linearly interpolated and the ends held, so every frame 1..N has exactly 6 rows (the GT always
     has the 6 players).
Output: out/pred/<tag>/<video>.txt, frame,court_x,court_y,team,jersey (no header).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
import idcls  # noqa: E402
from track import OUT_W, frames, video_path  # noqa: E402
from visualize import load_homography  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_tracks(path):
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2:6], a[:, 6]


def classify(split, video, frame_ids, boxes, model_path, shrink=1.0):
    """Probabilities (n_boxes, N_CLASSES) for every tracked box, in file order. With shrink < 1 the
    crops come from the same shrunken frames the detector saw (see track.py --shrink)."""
    net = idcls.load_net(model_path)
    rows_of = defaultdict(list)
    for i, f in enumerate(frame_ids):
        rows_of[f].append(i)
    probs = np.zeros((len(frame_ids), idcls.N_CLASSES), np.float32)
    pending_idx, pending_crops = [], []
    for n, img in enumerate(frames(video_path(split, video), shrink=shrink), 1):
        for i in rows_of.get(n, ()):
            pending_idx.append(i)
            pending_crops.append(idcls.crop(img, *(boxes[i] * shrink)))
        if len(pending_crops) >= 512:
            probs[pending_idx] = idcls.predict(net, np.stack(pending_crops))
            pending_idx, pending_crops = [], []
    if pending_crops:
        probs[pending_idx] = idcls.predict(net, np.stack(pending_crops))
    return probs


def roster_probs(probs, teams):
    """Columns = the 6 roster identities [(team, jersey)...] + other (non-roster mass folded in)."""
    rcls = idcls.roster_classes(teams)
    ids, cols = [], []
    for team, nums in teams.items():
        for n, c in zip(nums, rcls[team]):
            ids.append((team, n))
            cols.append(c)
    p = probs[:, cols]
    other = 1.0 - p.sum(1, keepdims=True)
    return np.hstack([p, other]), ids


def smooth_along_tracklets(frame_ids, track_ids, p, window=31):
    """Moving average of the box probabilities along each tracklet (robust to a few bad crops,
    and unlike a whole-tracklet mean it follows an identity switch inside a tracklet)."""
    q = p.copy()
    tracks = defaultdict(list)
    for i, t in enumerate(track_ids):
        tracks[t].append(i)
    for idx in tracks.values():
        idx = np.array(idx)[np.argsort(frame_ids[idx])]
        q[idx] = uniform_filter1d(p[idx], size=min(window, len(idx)), axis=0, mode="nearest")
    return q


def assign(frame_ids, q, n_ids, min_prob=0.2):
    """Return array (n_boxes,) with identity index or -1: per frame, Hungarian matching of boxes to
    the n_ids identities on the smoothed probabilities; weak or 'other'-dominated matches are dropped."""
    owner = np.full(len(frame_ids), -1)
    rows_of = defaultdict(list)
    for i, f in enumerate(frame_ids):
        rows_of[f].append(i)
    for idx in rows_of.values():
        s = q[idx, :n_ids]
        for r, k in zip(*linear_sum_assignment(-s)):
            if s[r, k] >= min_prob and s[r, k] > 0.5 * q[idx[r], n_ids]:
                owner[idx[r]] = k
    return owner


def court_positions(H, boxes):
    feet = np.stack([boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3]], 1)
    return cv2.perspectiveTransform(feet[None].astype(np.float64), H)[0]


def despike(xy, window=15, max_dev=60.0):
    """Mask of positions (time-ordered, one identity) within max_dev cm of their running median."""
    if len(xy) < 3:
        return np.ones(len(xy), bool)
    med = np.stack([median_filter(xy[:, d], window, mode="nearest") for d in (0, 1)], 1)
    return np.hypot(*(xy - med).T) < max_dev


def drop_short_segments(fr, xy, min_len=30, vmax=40.0, slack=50.0):
    """Mask of positions (time-ordered, one identity) outside short wrong-person stretches. The track is
    cut wherever a step is physically impossible (> slack + vmax cm per elapsed frame; 40 cm/frame = 12 m/s
    at 30 fps); a piece shorter than min_len observations is dropped when the pieces around it join up
    plausibly without it (or it is at an end, next to a longer piece). The median despike cannot catch
    these: a stretch of ~15 frames on the wrong player fills its own 15-observation window."""
    keep = np.ones(len(fr), bool)
    if len(fr) < 3 or min_len <= 0:
        return keep

    def joins(i, j):  # can the player get from observation i to observation j?
        return np.hypot(*(xy[j] - xy[i])) <= slack + vmax * (fr[j] - fr[i])

    cuts = [0] + [i + 1 for i in range(len(fr) - 1) if not joins(i, i + 1)] + [len(fr)]
    segs = [(a, b) for a, b in zip(cuts[:-1], cuts[1:])]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for k, (a, b) in enumerate(segs):
            if b - a >= min_len:
                continue
            prev = segs[k - 1] if k > 0 else None
            nxt = segs[k + 1] if k + 1 < len(segs) else None
            if prev and nxt:
                drop = joins(prev[1] - 1, nxt[0])
            else:
                other = prev or nxt
                drop = other[1] - other[0] > b - a
            if drop:
                keep[a:b] = False
                del segs[k]
                # re-merge neighbours that now join, so they count as one longer piece
                if prev and nxt and joins(prev[1] - 1, nxt[0]):
                    segs[k - 1:k + 1] = [(prev[0], nxt[1])]
                changed = True
                break
    return keep


def fill(frames_known, xy, n_frames):
    """Linear interpolation over 1..n_frames, ends held constant."""
    f = np.arange(1, n_frames + 1)
    if not len(frames_known):
        return np.tile([752.5, 552.5], (n_frames, 1))  # court centre
    return np.stack([np.interp(f, frames_known, xy[:, 0]), np.interp(f, frames_known, xy[:, 1])], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--split", default="train", choices=["train", "train720", "challenge"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "weights", "idcls_enc.pt"), help="identity classifier (tools/idcls.py train)")
    ap.add_argument("--out", help="prediction folder (default out/pred/<tag>)")
    ap.add_argument("--reclassify", action="store_true")
    ap.add_argument("--seg-min-len", type=int, default=30,
                    help="drop wrong-person (or mid-air) stretches shorter than this many observations (0 = off); chosen 28/10: no change on val, fixes 0114 B9 jump shot + 0115 A10 switch")
    args = ap.parse_args()

    data = os.path.join(ROOT, "data", args.split)
    rosters = idcls.load_team_rosters(os.path.join(data, "games_roster.csv"))
    H = load_homography(os.path.join(data, "court_registration.csv"))
    out_dir = args.out or os.path.join(ROOT, "out", "pred", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    for video in args.videos:
        tdir = os.path.join(ROOT, "out", "tracks", args.tag)
        meta = json.load(open(os.path.join(tdir, f"{video}.json")))
        frame_ids, track_ids, boxes, conf = load_tracks(os.path.join(tdir, f"{video}.txt"))
        cache = os.path.join(tdir, f"{video}.probs.npy")
        if args.reclassify or not os.path.exists(cache):
            np.save(cache, classify(args.split, video, frame_ids, boxes, args.model, meta.get("shrink", 1.0)))
        probs = np.load(cache)
        p, ids = roster_probs(probs, rosters[video])
        owner = assign(frame_ids, smooth_along_tracklets(frame_ids, track_ids, p), len(ids))

        # registration is in source-video pixels; boxes are 720p
        cap = cv2.VideoCapture(video_path(args.split, video))
        scale = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / OUT_W
        cap.release()
        xy = court_positions(H, boxes * scale)
        n = meta["frames"]
        tracks_out = []
        for k, (team, jersey) in enumerate(ids):
            m = np.where(owner == k)[0]
            m = m[np.argsort(frame_ids[m])]
            keep = despike(xy[m])
            keep[keep] = drop_short_segments(frame_ids[m][keep], xy[m][keep], args.seg_min_len)
            tracks_out.append(fill(frame_ids[m][keep], xy[m][keep], n))
            print(f"  {team}{jersey}: {keep.sum()} boxes ({keep.sum() / n:.0%} of frames), {(~keep).sum()} spikes dropped")
        with open(os.path.join(out_dir, f"{video}.txt"), "w") as f:
            for i in range(n):
                for (team, jersey), tr in zip(ids, tracks_out):
                    f.write(f"{i + 1},{tr[i, 0]:.3f},{tr[i, 1]:.3f},{team},{jersey}\n")
        print(f"{video}: {n} frames, {len(np.unique(track_ids))} tracklets, "
              f"{(owner >= 0).mean():.0%} of boxes assigned -> {out_dir}/{video}.txt")


if __name__ == "__main__":
    main()
