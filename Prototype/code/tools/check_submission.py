"""Check prediction files against the submission rules before uploading.

    .venv/bin/python tools/check_submission.py --pred out/pred/final --split challenge

For every video of the split's roster: the file <video>.txt exists, has no header, every row is
frame_id,court_x,court_y,team_id,jersey_number with finite coordinates, team_id in {A, B}, the frames
are exactly 1..N where N is the number of frames decoded from the video (read sequentially, the
container's frame count can be wrong), and every frame has the 6 roster identities once each.
"""
import argparse
import math
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
import idcls  # noqa: E402
from track import video_path  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def count_frames(path):
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.grab():
        n += 1
    return n


def check(path, roster, n_frames):
    want = {(team, n) for team, nums in roster.items() for n in nums}
    seen = {}
    for k, line in enumerate(open(path), 1):
        p = line.strip().split(",")
        if len(p) != 5:
            return f"line {k}: expected 5 fields, got {line.strip()!r}"
        try:
            f, x, y, team, jersey = int(p[0]), float(p[1]), float(p[2]), p[3], int(p[4])
        except ValueError:
            return f"line {k}: not frame,x,y,team,jersey: {line.strip()!r}"
        if not (math.isfinite(x) and math.isfinite(y)) or team not in ("A", "B"):
            return f"line {k}: bad value {line.strip()!r}"
        seen.setdefault(f, []).append((team, jersey))
    if sorted(seen) != list(range(1, n_frames + 1)):
        return f"frames {min(seen)}..{max(seen)} ({len(seen)} distinct), video has 1..{n_frames}"
    bad = [f for f, ids in seen.items() if len(ids) != len(want) or set(ids) != want]
    if bad:
        return f"{len(bad)} frames without exactly the 6 roster identities, first {bad[0]}: {seen[bad[0]]}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="challenge")
    args = ap.parse_args()
    rosters = idcls.load_team_rosters(os.path.join(ROOT, "data", args.split, "games_roster.csv"))
    ok = True
    for video, roster in sorted(rosters.items()):
        path = os.path.join(args.pred, f"{video}.txt")
        if not os.path.exists(path):
            print(f"{video}: MISSING {path}")
            ok = False
            continue
        n = count_frames(video_path(args.split, video))
        err = check(path, roster, n)
        ok &= err is None
        print(f"{video}: {'OK' if err is None else 'FAIL ' + err} ({n} frames x {sum(map(len, roster.values()))} rows)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
