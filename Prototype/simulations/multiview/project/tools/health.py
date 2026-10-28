"""Label-free health signals of a run, comparable between val (has GT) and Challenge (no GT).

    .venv/bin/python tools/health.py --split challenge --tags ch_r2_last_c2 ch_r2_botsort --videos IMG_0114 IMG_0115

Per video: player tracklets (tracklets mostly assigned to a roster identity: fewer = fewer breaks),
assigned boxes, uncertain crops (top roster probability < 0.5), and in the final output the number of
impossible jumps (> 1 m in one frame) and the share of identity-frames moving faster than 8 m/s.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import idcls  # noqa: E402
from eval_tihota import load_frames  # noqa: E402
from pipeline import assign, load_tracks, roster_probs, smooth_along_tracklets  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="challenge")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    args = ap.parse_args()
    rosters = idcls.load_team_rosters(os.path.join(ROOT, "data", args.split, "games_roster.csv"))
    for tag in args.tags:
        for v in args.videos:
            fid, tid, _, _ = load_tracks(os.path.join(ROOT, "out", "tracks", tag, f"{v}.txt"))
            p, ids = roster_probs(np.load(os.path.join(ROOT, "out", "tracks", tag, f"{v}.probs.npy")), rosters[v])
            owner = assign(fid, smooth_along_tracklets(fid, tid, p), len(ids))
            a = owner >= 0
            players = sum(a[tid == t].mean() > 0.5 for t in np.unique(tid))
            pred = load_frames(os.path.join(ROOT, "out", "pred", tag, f"{v}.txt"))
            pos = np.stack([pred[f][0] for f in sorted(pred)])
            step = np.hypot(*np.diff(pos, axis=0).transpose(2, 0, 1))  # cm per frame
            print(f"{tag:16s} {v}  player tracklets {players:4d}  assigned {a.mean():6.1%}  "
                  f"uncertain crops {np.mean(p[:, :len(ids)].max(1) < 0.5):5.1%}  "
                  f"jumps>1m {int((step > 100).sum()):3d}  >8m/s {np.mean(step * 0.3 > 8):.2%}")


if __name__ == "__main__":
    main()
