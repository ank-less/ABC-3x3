"""Render player annotations on a training video next to a 3x3 court minimap.

Usage:
    .venv/bin/python tools/visualize.py IMG_0106                      # whole video -> out/IMG_0106_vis.mp4
    .venv/bin/python tools/visualize.py IMG_0106 --start 300 --end 900
    .venv/bin/python tools/visualize.py IMG_0106 --start 500 --out out/f500.png   # single frame

Reads the clean dataset from tools/prepare_data.py (data/train).
Frame numbers are the 1-based annotation frame numbers (annotation frame 1 = first video frame).
Court positions are computed from each box's bottom-centre with the homography in
court_registration.csv; this reproduces the provided "court position" files exactly and also
works for videos that have no such file.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np

COURT_W, COURT_H = 1505.0, 1105.0  # cm; FIBA 3x3 half court, basket on the court_y = 0 baseline
MARGIN = 150.0  # cm of out-of-bounds area shown around the court on the minimap

TEAM_COLORS = {"A": (40, 140, 255), "B": (230, 200, 30), "?": (160, 160, 160)}  # BGR


def load_homography(path):
    rows = list(csv.DictReader(open(path)))
    img = np.array([[float(r["image_x"]), float(r["image_y"])] for r in rows], np.float32)
    court = np.array([[float(r["court_x"]), float(r["court_y"])] for r in rows], np.float32)
    return cv2.getPerspectiveTransform(img, court)


def load_roster(path, video):
    """Return game_start_frame and {track_id: (team, jersey, role)} for one Training video."""
    row = next(r for r in csv.DictReader(open(path)) if r["video"] == video)
    jersey = {int(k): v.strip() for k, v in (p.split(":") for p in row["track_id: jersey"].split(","))}
    offense = [int(t) for t in row["offense_track_ids"].split(",")]
    defense = [int(t) for t in row["defense_track_ids"].split(",")]
    teams = {}
    for part in row["team: jersey"].split(";"):
        name, nums = part.split(":")
        teams[name.strip()] = sorted(n.strip() for n in nums.split(","))
    # Jersey numbers repeat across teams (e.g. both teams wear 9), so the team is decided by
    # which roster matches the offense group as a whole.
    off_jerseys = sorted(jersey[t] for t in offense)
    off_team = next((name for name, nums in teams.items() if nums == off_jerseys), None)
    if off_team is None:
        print(f"warning: offense jerseys {off_jerseys} match no team in {teams}", file=sys.stderr)
    def_team = next((name for name in teams if name != off_team), None)
    players = {}
    for t in offense:
        players[t] = (off_team or "?", jersey[t], "OFF")
    for t in defense:
        players[t] = (def_team or "?", jersey[t], "DEF")
    return int(row["game_start_frame"]), players


def load_boxes(path):
    """Return {frame: {track_id: (x, y, w, h, conf)}}, keeping the most confident box on duplicates."""
    boxes = defaultdict(dict)
    for line in open(path):
        p = line.strip().split(",")
        frame, tid = int(p[0]), int(p[1])
        box = tuple(map(float, p[2:7]))
        if tid not in boxes[frame] or box[4] > boxes[frame][tid][4]:
            boxes[frame][tid] = box
    return boxes


def to_court(H, boxes):
    """Project the bottom-centre of each box to court coordinates."""
    if not boxes:
        return {}
    ids = list(boxes)
    feet = np.array([[[b[0] + b[2] / 2, b[1] + b[3]] for b in boxes.values()]], np.float32)
    pts = cv2.perspectiveTransform(feet, H)[0]
    return dict(zip(ids, map(tuple, pts)))


class Minimap:
    """Top-down 3x3 court, oriented like the camera view: basket at the top, court_x growing leftwards."""

    def __init__(self, height):
        self.s = height / (COURT_H + 2 * MARGIN)
        self.w = int(round((COURT_W + 2 * MARGIN) * self.s)) // 2 * 2  # even, for video encoders
        self.h = height
        self.base = self._draw_court()

    def px(self, cx, cy):
        x = (COURT_W - cx + MARGIN) * self.s
        y = (cy + MARGIN) * self.s
        return int(round(x)), int(round(y))

    def _draw_court(self):
        img = np.full((self.h, self.w, 3), (70, 70, 70), np.uint8)
        line = (255, 255, 255)
        t = max(1, int(round(5 * self.s)))
        cv2.rectangle(img, self.px(COURT_W, 0), self.px(0, COURT_H), (160, 90, 40), -1)
        cv2.rectangle(img, self.px(COURT_W, 0), self.px(0, COURT_H), line, t)
        cx = COURT_W / 2
        # Painted key: 4.9 m wide, free-throw line 5.8 m from the baseline.
        cv2.rectangle(img, self.px(cx + 245, 0), self.px(cx - 245, 580), (205, 205, 205), -1)
        cv2.rectangle(img, self.px(cx + 245, 0), self.px(cx - 245, 580), line, t)
        # Basket 1.575 m from the baseline; 2-point arc r = 6.75 m with straight lines 0.9 m in from the sidelines.
        by, r = 157.5, 675.0
        side = 90.0
        dy = (r * r - (cx - side) ** 2) ** 0.5
        cv2.line(img, self.px(side, 0), self.px(side, by + dy), line, t)
        cv2.line(img, self.px(COURT_W - side, 0), self.px(COURT_W - side, by + dy), line, t)
        arc = [self.px(cx + r * np.cos(a), by + r * np.sin(a))
               for a in np.linspace(np.arccos((side - cx) / r), np.arccos((COURT_W - side - cx) / r), 64)]
        cv2.polylines(img, [np.array(arc, np.int32)], False, line, t, cv2.LINE_AA)
        # Backboard (1.2 m from the baseline, 1.8 m wide) and rim.
        cv2.line(img, self.px(cx + 90, 120), self.px(cx - 90, 120), line, t)
        cv2.circle(img, self.px(cx, by), max(2, int(round(23 * self.s))), (0, 90, 255), t, cv2.LINE_AA)
        # Legend in the bottom margin.
        scale = self.h / 1400
        y = self.h - int(MARGIN * self.s / 3)
        x = int(MARGIN * self.s)
        for team in ("A", "B"):
            cv2.circle(img, (x, y - int(8 * scale)), int(10 * scale), TEAM_COLORS[team], -1, cv2.LINE_AA)
            put_text(img, f"team {team}", (x + int(18 * scale), y), scale)
            x += int(170 * scale)
        put_text(img, "ring: black=OFF white=DEF", (x, y), scale)
        return img

    def render(self, positions, trails, players):
        img = self.base.copy()
        r = max(4, int(round(28 * self.s)))
        for tid, trail in trails.items():
            if len(trail) > 1:
                color = TEAM_COLORS[players.get(tid, ("?",))[0]]
                pts = np.array([self.px(*p) for p in trail], np.int32)
                cv2.polylines(img, [pts], False, color, max(1, r // 4), cv2.LINE_AA)
        for tid, (cx, cy) in positions.items():
            team, jersey, role = players.get(tid, ("?", "?", ""))
            color = TEAM_COLORS[team]
            # Clamp to the visible panel; players outside it are drawn hollow at the edge.
            inside = -MARGIN <= cx <= COURT_W + MARGIN and -MARGIN <= cy <= COURT_H + MARGIN
            x, y = self.px(min(max(cx, -MARGIN), COURT_W + MARGIN), min(max(cy, -MARGIN), COURT_H + MARGIN))
            cv2.circle(img, (x, y), r, color, -1 if inside else 2, cv2.LINE_AA)
            cv2.circle(img, (x, y), r, (0, 0, 0) if role == "OFF" else (255, 255, 255), max(1, r // 6), cv2.LINE_AA)
            put_text(img, jersey, (x, y), r / 22, center=True)
        return img


def put_text(img, text, org, scale, color=(255, 255, 255), center=False):
    thick = max(1, int(round(scale * 2)))
    if center:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = (org[0] - w // 2, org[1] + h // 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_boxes(frame, boxes, players, k):
    """Draw boxes on a frame already resized by factor k."""
    t = max(1, int(round(6 * k)))
    for tid, (x, y, w, h, conf) in boxes.items():
        team, jersey, role = players.get(tid, ("?", "?", ""))
        color = TEAM_COLORS[team]
        p1, p2 = (int(x * k), int(y * k)), (int((x + w) * k), int((y + h) * k))
        cv2.rectangle(frame, p1, p2, color, t)
        label = f"T{tid} {team}#{jersey} {role}"
        scale = 1.4 * k
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - base - 2), (p1[0] + tw + 2, p1[1]), color, -1)
        cv2.putText(frame, label, (p1[0] + 1, p1[1] - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video name, e.g. IMG_0106")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "train"),
                    help="clean dataset built by tools/prepare_data.py")
    ap.add_argument("--start", type=int, default=1, help="first annotation frame (1-based)")
    ap.add_argument("--end", type=int, default=None, help="last frame, exclusive (default: end of annotations)")
    ap.add_argument("--height", type=int, default=720, help="output height in pixels")
    ap.add_argument("--trail", type=int, default=45, help="minimap trail length in frames (0 = off)")
    ap.add_argument("--out", default=None, help=".mp4 for a clip, .png/.jpg for the single --start frame")
    args = ap.parse_args()

    video_path = os.path.join(args.data, "videos", f"{args.video}.MOV")
    if not os.path.exists(video_path):
        sys.exit(f"{video_path} not found (IMG_0107 is missing from the download)")
    H = load_homography(os.path.join(args.data, "court_registration.csv"))
    game_start, players = load_roster(os.path.join(args.data, "games_roster.csv"), args.video)
    boxes = load_boxes(os.path.join(args.data, "bboxes", f"{args.video}.txt"))
    end = args.end or max(boxes) + 1
    out = args.out or os.path.join("out", f"{args.video}_vis.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    single = out.lower().endswith((".png", ".jpg", ".jpeg"))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    k = args.height / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vid_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * k))
    minimap = Minimap(args.height)
    trails = defaultdict(lambda: deque(maxlen=max(1, args.trail)))
    if args.trail and args.start > 1:
        # Warm up the trails so the first rendered frame already shows recent movement.
        for f in range(max(1, args.start - args.trail), args.start):
            for tid, p in to_court(H, boxes.get(f, {})).items():
                trails[tid].append(p)

    # Skip ahead by decoding, not CAP_PROP_POS_FRAMES: IMG_0104's container lists 11400 frames but only
    # 11376 decode, and timestamp-based seeking there lands up to 3 frames off the annotation index.
    for _ in range(args.start - 1):
        cap.grab()
    writer = None
    if not single:
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w + minimap.w, args.height))
    for f in range(args.start, end):
        ok, frame = cap.read()
        if not ok:
            print(f"video ended at frame {f - 1}", file=sys.stderr)
            break
        frame = cv2.resize(frame, (vid_w, args.height), interpolation=cv2.INTER_AREA)
        fboxes = boxes.get(f, {})
        draw_boxes(frame, fboxes, players, k)
        state = "GAME" if f >= game_start else f"PRE-GAME (starts at {game_start})"
        put_text(frame, f"{args.video}  frame {f}  {(f - 1) / fps:6.1f}s  {state}", (12, 36), 0.9)

        positions = to_court(H, fboxes)
        for tid, p in positions.items():
            if args.trail:
                trails[tid].append(p)
        panel = minimap.render(positions, trails if args.trail else {}, players)
        canvas = np.hstack([frame, panel])
        if single:
            cv2.imwrite(out, canvas)
            break
        writer.write(canvas)
        if (f - args.start) % 500 == 0:
            print(f"frame {f}/{end - 1}", file=sys.stderr)
    if writer:
        writer.release()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
