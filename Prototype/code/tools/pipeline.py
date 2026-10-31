"""Stages 2-3: identify tracked boxes and write a submission file per video.

    .venv/bin/python tools/pipeline.py IMG_0108 IMG_0109 --tag coco_l          # Training val videos
    .venv/bin/python tools/pipeline.py IMG_0114 IMG_0115 --split challenge --tag ft_l
    .venv/bin/python tools/eval_tihota.py --pred out/pred/coco_l --videos IMG_0108 IMG_0109

Input: out/tracks/<tag>/<video>.txt from tools/track.py (720p boxes).
  1. classify: every box crop -> probabilities over the 12 (colour, number) identities + other
     (tools/idcls.py), cached as out/tracks/<tag>/<video>.probs.npy;
  2. restrict to the video's roster: the 6 roster identities, everything else counts as "other";
  3. box -> identity: average probabilities in a centered 31-frame window, split at gaps over 3 frames,
     then per-frame Hungarian-match boxes to the 6 identities (one box per identity per frame);
  4. court position = homography(bottom-centre of the box); a local Jacobian estimates projection
     sensitivity. Time-based 15-frame despiking uses a capped uncertainty-aware threshold, then short
     physically impossible stretches are dropped; gaps up to 30 frames are interpolated and longer gaps
     use the nearest measured endpoint, so every frame 1..N still has exactly 6 rows.
Output: out/pred/<tag>/<video>.txt, frame,court_x,court_y,team,jersey (no header).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
import idcls  # noqa: E402
from track import OUT_H, OUT_W, frames, shrink_scales, video_path  # noqa: E402
from visualize import load_homography  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_tracks(path):
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2:6], a[:, 6]


def classify(split, video, frame_ids, boxes, model_path, shrink=1.0):
    """Probabilities (n_boxes, N_CLASSES) for every tracked box, in file order. With shrink < 1 the
    crops come from the same shrunken frames the detector saw (see track.py --shrink)."""
    net = idcls.load_net(model_path)
    sx, sy = shrink_scales(shrink)
    crop_scale = np.array([sx, sy, sx, sy], dtype=np.float64)
    rows_of = defaultdict(list)
    for i, f in enumerate(frame_ids):
        rows_of[f].append(i)
    probs = np.zeros((len(frame_ids), idcls.N_CLASSES), np.float32)
    pending_idx, pending_crops = [], []
    for n, img in enumerate(frames(video_path(split, video), shrink=shrink), 1):
        for i in rows_of.get(n, ()):
            pending_idx.append(i)
            pending_crops.append(idcls.crop(img, *(boxes[i] * crop_scale)))
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


def _split_indices_by_gap(indices, frame_ids, max_gap):
    """Split sorted observation indices where successive observations are too far apart in time."""
    if len(indices) == 0:
        return []
    cuts = np.flatnonzero(np.diff(frame_ids[indices]) > max_gap) + 1
    bounds = np.r_[0, cuts, len(indices)]
    return [indices[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def smooth_along_tracklets(frame_ids, track_ids, p, window=31, max_gap=3):
    """Average probabilities in a centered *frame* window, without bridging long gaps.

    Duplicate detections for one track in the same frame are averaged first, so they cannot
    overweight a class. The track ID is only a tracklet hint; a gap over ``max_gap`` starts a
    fresh smoothing segment to avoid carrying appearance through an occlusion or stale tracker state.
    """
    q = p.copy()
    tracks = defaultdict(list)
    for i, t in enumerate(track_ids):
        tracks[t].append(i)
    left = (window - 1) // 2
    right = window // 2
    for track_idx in tracks.values():
        idx = np.asarray(track_idx, dtype=int)
        idx = np.array(idx)[np.argsort(frame_ids[idx])]
        for segment in _split_indices_by_gap(idx, frame_ids, max_gap):
            segment_frames, inverse = np.unique(frame_ids[segment], return_inverse=True)
            frame_sums = np.zeros((len(segment_frames), p.shape[1]), dtype=np.float64)
            np.add.at(frame_sums, inverse, p[segment])
            counts = np.bincount(inverse).astype(np.float64)[:, None]
            frame_probs = frame_sums / counts
            prefix = np.vstack([np.zeros((1, p.shape[1])), np.cumsum(frame_probs, axis=0)])
            smoothed = np.empty_like(frame_probs)
            for j, frame in enumerate(segment_frames):
                lo = np.searchsorted(segment_frames, frame - left, side="left")
                hi = np.searchsorted(segment_frames, frame + right, side="right")
                smoothed[j] = (prefix[hi] - prefix[lo]) / (hi - lo)
            q[segment] = smoothed[inverse]
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


def box_bottom_centres(boxes):
    """The competition's Training labels use the bottom-centre of each annotated box."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.stack([boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3]], axis=1)


def projection_uncertainty(H, boxes, confidence=None):
    """Propagate approximate box-bottom pixel uncertainty through the local homography Jacobian.

    This is a local sensitivity estimate, not a 3-D correction. A court-plane homography cannot
    recover a true foot point when a player is airborne or the box bottom is occluded.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    x, y = box_bottom_centres(boxes).T
    den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    safe_den = np.where(np.abs(den) > 1e-9, den, np.nan)
    u_num = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    v_num = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    u = u_num / safe_den
    v = v_num / safe_den
    den_sq = safe_den ** 2
    jac = np.empty((len(boxes), 2, 2), dtype=np.float64)
    jac[:, 0, 0] = (H[0, 0] * safe_den - H[2, 0] * u_num) / den_sq
    jac[:, 0, 1] = (H[0, 1] * safe_den - H[2, 1] * u_num) / den_sq
    jac[:, 1, 0] = (H[1, 0] * safe_den - H[2, 0] * v_num) / den_sq
    jac[:, 1, 1] = (H[1, 1] * safe_den - H[2, 1] * v_num) / den_sq

    sigma_x = np.maximum(1.5, 0.05 * np.maximum(boxes[:, 2], 1.0))
    sigma_y = np.maximum(2.0, 0.04 * np.maximum(boxes[:, 3], 1.0))
    if confidence is not None:
        conf = np.clip(np.asarray(confidence, dtype=np.float64), 0.1, 1.0)
        quality_scale = np.minimum(2.0, 1.0 / np.sqrt(conf))
        sigma_x *= quality_scale
        sigma_y *= quality_scale
    weighted_jac = jac * np.stack([sigma_x, sigma_y], axis=1)[:, None, :]
    sigma_cm = np.sqrt(np.sum(weighted_jac ** 2, axis=(1, 2)))
    sigma_cm[~np.isfinite(u) | ~np.isfinite(v)] = np.inf
    return sigma_cm


def court_positions(H, boxes, confidence=None, return_uncertainty=False):
    feet = box_bottom_centres(boxes)
    if len(feet) == 0:
        xy = np.empty((0, 2), dtype=np.float64)
    else:
        xy = cv2.perspectiveTransform(feet[None].astype(np.float64), H)[0]
    if return_uncertainty:
        return xy, projection_uncertainty(H, boxes, confidence)
    return xy


def despike(xy, frame_ids=None, uncertainty_cm=None, window=15, max_dev=60.0,
            max_dev_cap=120.0, max_uncertainty_cm=80.0, max_gap=3):
    """Reject local position spikes using actual frame time and projection sensitivity.

    ``window`` is measured in frames, not observations. The adaptive tolerance accounts for local
    homography amplification but is capped; highly uncertain projections are rejected outright.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = len(xy)
    keep = np.isfinite(xy).all(axis=1)
    if frame_ids is None:
        frame_ids = np.arange(n, dtype=int)
    frame_ids = np.asarray(frame_ids, dtype=int)
    if uncertainty_cm is not None:
        uncertainty_cm = np.asarray(uncertainty_cm, dtype=np.float64)
        keep &= np.isfinite(uncertainty_cm) & (uncertainty_cm <= max_uncertainty_cm)
    if n < 3:
        return keep

    radius_left = (window - 1) // 2
    radius_right = window // 2
    order = np.argsort(frame_ids, kind="stable")
    sorted_frames, sorted_xy = frame_ids[order], xy[order]
    sorted_uncertainty = uncertainty_cm[order] if uncertainty_cm is not None else None
    sorted_keep = keep[order].copy()
    for segment in _split_indices_by_gap(np.arange(n), sorted_frames, max_gap):
        segment_frames = sorted_frames[segment]
        for row_i in segment:
            lo = np.searchsorted(segment_frames, sorted_frames[row_i] - radius_left, side="left")
            hi = np.searchsorted(segment_frames, sorted_frames[row_i] + radius_right, side="right")
            neighbours = segment[lo:hi]
            neighbours = neighbours[sorted_keep[neighbours]]
            if len(neighbours) < 3 or not sorted_keep[row_i]:
                continue
            median = np.median(sorted_xy[neighbours], axis=0)
            tolerance = max_dev
            if sorted_uncertainty is not None:
                tolerance = min(max_dev_cap, max_dev + 2.5 * sorted_uncertainty[row_i])
            if np.linalg.norm(sorted_xy[row_i] - median) > tolerance:
                sorted_keep[row_i] = False
    keep[order] = sorted_keep
    return keep


def drop_short_segments(fr, xy, min_len=30, vmax=40.0, slack=50.0):
    """Mask of positions (time-ordered, one identity) outside short wrong-person stretches. The track is
    cut wherever a step is physically impossible (> slack + vmax cm per elapsed frame; 40 cm/frame = 12 m/s
    at 30 fps); a piece shorter than min_len elapsed frames is dropped when the pieces around it join up
    plausibly without it (or it is at an end, next to a longer piece)."""
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
            duration_frames = fr[b - 1] - fr[a] + 1
            if duration_frames >= min_len:
                continue
            prev = segs[k - 1] if k > 0 else None
            nxt = segs[k + 1] if k + 1 < len(segs) else None
            if prev and nxt:
                drop = joins(prev[1] - 1, nxt[0])
            else:
                other = prev or nxt
                other_duration = fr[other[1] - 1] - fr[other[0]] + 1
                drop = other_duration > duration_frames
            if drop:
                keep[a:b] = False
                del segs[k]
                # re-merge neighbours that now join, so they count as one longer piece
                if prev and nxt and joins(prev[1] - 1, nxt[0]):
                    segs[k - 1:k + 1] = [(prev[0], nxt[1])]
                changed = True
                break
    return keep


def fill(frames_known, xy, n_frames, max_interp_gap=30):
    """Interpolate short gaps; use the nearest measured endpoint across longer unsupported gaps.

    Endpoint positions remain held constant. This keeps the required dense output without drawing a
    potentially misleading straight path across long occlusions or tracker/identity losses.
    """
    f = np.arange(1, n_frames + 1)
    if not len(frames_known):
        return np.tile([752.5, 552.5], (n_frames, 1))  # court centre
    frames_known = np.asarray(frames_known, dtype=int)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    order = np.argsort(frames_known, kind="stable")
    frames_known, xy = frames_known[order], xy[order]
    unique_frames, inverse = np.unique(frames_known, return_inverse=True)
    if len(unique_frames) != len(frames_known):
        sums = np.zeros((len(unique_frames), 2), dtype=np.float64)
        counts = np.bincount(inverse).astype(np.float64)[:, None]
        np.add.at(sums, inverse, xy)
        frames_known, xy = unique_frames, sums / counts

    right = np.searchsorted(frames_known, f, side="right")
    left = np.clip(right - 1, 0, len(frames_known) - 1)
    right_idx = np.clip(right, 0, len(frames_known) - 1)
    out = xy[left].copy()
    interior = (right > 0) & (right < len(frames_known)) & (frames_known[left] < f)
    gaps = frames_known[right_idx] - frames_known[left]
    short = interior & (gaps <= max_interp_gap)
    ratio = np.zeros(len(f), dtype=np.float64)
    ratio[short] = (f[short] - frames_known[left[short]]) / gaps[short]
    out[short] = xy[left[short]] + ratio[short, None] * (xy[right_idx[short]] - xy[left[short]])
    long = interior & ~short
    use_right = long & ((frames_known[right_idx] - f) < (f - frames_known[left]))
    out[use_right] = xy[right_idx[use_right]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--split", default="train", choices=["train", "train720", "challenge"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "weights", "idcls_enc.pt"), help="identity classifier (tools/idcls.py train)")
    ap.add_argument("--out", help="prediction folder (default out/pred/<tag>)")
    ap.add_argument("--reclassify", action="store_true")
    ap.add_argument("--seg-min-len", type=int, default=30,
                    help="drop wrong-person (or mid-air) stretches shorter than this many elapsed frames (0 = off)")
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
        source_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        source_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        if source_width <= 0 or source_height <= 0:
            raise RuntimeError(f"could not read source dimensions for {video}")
        box_scale = np.array([source_width / OUT_W, source_height / OUT_H,
                              source_width / OUT_W, source_height / OUT_H], dtype=np.float64)
        source_boxes = boxes * box_scale
        xy, sigma_cm = court_positions(H, source_boxes, conf, return_uncertainty=True)
        n = meta["frames"]
        tracks_out = []
        for k, (team, jersey) in enumerate(ids):
            m = np.where(owner == k)[0]
            m = m[np.argsort(frame_ids[m])]
            keep = despike(xy[m], frame_ids[m], sigma_cm[m])
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
