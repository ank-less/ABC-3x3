#!/usr/bin/env python3
"""Create labeled 2D virtual-camera stress clips from one labeled Training clip.

The planar homography is exact for court points. Applying it to the full image is a
deliberately approximate view simulation: players are 3D and will also be warped.
"""
import argparse
import csv
import json
import os
import shutil

import cv2
import numpy as np


W, H = 1280, 720
PROFILES = [
    # name, yaw, pitch, roll, camera_dx_m, camera_dy_m, plane_distance_m
    ("base", 0, 0, 0, 0.0, 0.0, 18.0),
    ("yaw_left_10", -10, 0, 0, -0.35, 0.0, 18.0),
    ("yaw_right_10", 10, 0, 0, 0.35, 0.0, 18.0),
    ("high_left_18", -18, 8, -2, -0.8, 0.3, 17.0),
    ("low_right_18", 18, -8, 2, 0.8, -0.3, 17.0),
    ("hard_diagonal_left", -27, 13, -4, -1.4, 0.7, 16.0),
    ("hard_diagonal_right", 27, -13, 4, 1.4, -0.7, 16.0),
    ("extreme_oblique", 34, 18, 6, 2.0, 1.0, 15.0),
]


def rotation(yaw, pitch, roll):
    y, p, r = np.deg2rad([yaw, pitch, roll])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def camera_warp(yaw, pitch, roll, dx, dy, distance, anchor):
    # Plane-induced homography for an assumed 60-degree horizontal field of view.
    f = W / (2 * np.tan(np.deg2rad(60) / 2))
    k = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)
    r = rotation(yaw, pitch, roll)
    t = np.array([[dx], [dy], [0.0]], dtype=np.float64)
    normal = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    m = k @ (r + t @ normal / distance) @ np.linalg.inv(k)
    # Aim each virtual camera back at the court centre so the difficult views still contain play.
    moved = transform_point(m, anchor)
    recenter = np.array([[1.0, 0.0, anchor[0] - moved[0]],
                         [0.0, 1.0, anchor[1] - moved[1]],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    m = recenter @ m
    return m / m[2, 2]


def transform_point(m, xy):
    p = np.array([[[xy[0], xy[1]]]], dtype=np.float64)
    return cv2.perspectiveTransform(p, m)[0, 0]


def write_registration(path, raw_registration, scale, m):
    rows = list(csv.DictReader(open(raw_registration, newline="")))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_id", "image_x", "image_y", "court_x", "court_y"])
        for row in rows:
            x = float(row["image_x"]) * scale
            y = float(row["image_y"]) * scale
            xp, yp = transform_point(m, (x, y))
            writer.writerow([row["point_id"], f"{xp:.6f}", f"{yp:.6f}", row["court_x"], row["court_y"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-video", required=True, help="Challenge-style 1280x720 Training clip")
    ap.add_argument("--source-width", type=int, default=3840, help="width used by the registration CSV")
    ap.add_argument("--registration", required=True, help="Training court_registration.csv")
    ap.add_argument("--roster", required=True, help="Training games_roster.csv")
    ap.add_argument("--ground-truth", required=True, help="original Training court-position file")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--out", required=True, help="isolated simulation project root")
    args = ap.parse_args()

    data = os.path.join(args.out, "data", "train")
    videos = os.path.join(data, "videos")
    gt_dir = os.path.join(data, "court")
    meta_dir = os.path.join(args.out, "metadata")
    preview_dir = os.path.join(args.out, "previews")
    for directory in (videos, gt_dir, meta_dir, preview_dir):
        os.makedirs(directory, exist_ok=True)

    scale = W / args.source_width
    reg_rows = list(csv.DictReader(open(args.registration, newline="")))
    source_corners = [(float(r["image_x"]) * scale, float(r["image_y"]) * scale) for r in reg_rows]
    court_anchor = np.mean(source_corners, axis=0)
    cap = cv2.VideoCapture(args.base_video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open base video: {args.base_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(args.frames, source_frames)
    if n_frames <= 0:
        raise SystemExit("Base video has no decodable frames")

    src_roster = next(r for r in csv.DictReader(open(args.roster, newline="")) if r["video"] == "IMG_0108")
    roster_path = os.path.join(data, "games_roster.csv")
    roster_rows = []

    for name, yaw, pitch, roll, dx, dy, distance in PROFILES:
        video_id = f"SIM0108_{name}"
        out_video = os.path.join(videos, f"{video_id}.mp4")
        out_gt = os.path.join(gt_dir, f"{video_id}.txt")
        m = camera_warp(yaw, pitch, roll, dx, dy, distance, court_anchor)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not writer.isOpened():
            raise SystemExit(f"Cannot write {out_video}")
        first = None
        written = 0
        while written < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            warped = cv2.warpPerspective(frame, m, (W, H), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(90, 90, 90))
            writer.write(warped)
            if first is None:
                first = warped
            written += 1
        writer.release()
        if written != n_frames:
            raise SystemExit(f"{video_id}: expected {n_frames} frames, wrote {written}")
        cv2.imwrite(os.path.join(preview_dir, f"{video_id}.jpg"), first, [cv2.IMWRITE_JPEG_QUALITY, 92])

        registration = os.path.join(meta_dir, f"{video_id}_court_registration.csv")
        write_registration(registration, args.registration, scale, m)
        with open(out_gt, "w") as f:
            for line in open(args.ground_truth):
                fields = line.strip().split(",")
                if fields and 1 <= int(fields[0]) <= written:
                    f.write(",".join(fields) + "\n")
        roster_row = dict(src_roster)
        roster_row["video"] = video_id
        roster_rows.append(roster_row)
        with open(os.path.join(meta_dir, f"{video_id}.json"), "w") as f:
            json.dump({"video": video_id, "profile": name, "yaw_deg": yaw, "pitch_deg": pitch,
                       "roll_deg": roll, "translation_m": [dx, dy], "distance_m": distance,
                       "frames": written, "fps": fps, "warp": m.tolist()}, f, indent=2)
        corners = [transform_point(m, (float(r["image_x"]) * scale, float(r["image_y"]) * scale)).tolist()
                   for r in reg_rows]
        print(f"{video_id}: {written} frames, yaw={yaw:+} pitch={pitch:+} roll={roll:+}; court corners={np.round(corners).astype(int).tolist()}")

    cap.release()
    with open(roster_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(src_roster))
        writer.writeheader()
        writer.writerows(roster_rows)

    print(f"Generated {len(PROFILES)} views from {n_frames} Training frames in {args.out}")


if __name__ == "__main__":
    main()
