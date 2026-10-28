"""Figures for the paper (run from the project root after ./run_challenge.sh and ./run_val.sh inputs exist).

    .venv/bin/python paper/make_figures.py        # -> paper/figures/*.pdf / *.jpg
"""
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "tools")
import idcls  # noqa: E402
import pipeline  # noqa: E402
from eval_tihota import load_frames  # noqa: E402
from track import video_path  # noqa: E402
from visualize import load_homography  # noqa: E402

OUT = "paper/figures"
COLOURS = {"purple": "#7b3fa0", "yellow": "#d4a017", "green": "#2e9e44", "pink": "#e75480"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})


def read_frame(path, n):
    cap = cv2.VideoCapture(path)
    for _ in range(n):
        ok, img = cap.read()
    return img


def identities(split, tag, video):
    fr, tid, boxes, _ = pipeline.load_tracks(f"out/tracks/{tag}/{video}.txt")
    rost = idcls.load_team_rosters(f"data/{split}/games_roster.csv")[video]
    p, ids = pipeline.roster_probs(np.load(f"out/tracks/{tag}/{video}.probs.npy"), rost)
    owner = pipeline.assign(fr, pipeline.smooth_along_tracklets(fr, tid, p), len(ids))
    colour = {team: idcls.colour_of(nums) for team, nums in rost.items()}
    return fr, boxes, owner, ids, colour


def draw(ax, img, fr, boxes, owner, ids, colour, f):
    ax.imshow(img[..., ::-1])
    for i in np.where(fr == f)[0]:
        x, y, w, h = boxes[i]
        k = owner[i]
        c = COLOURS[colour[ids[k][0]]] if k >= 0 else "#bbbbbb"
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=0.8, ec=c, ls="-" if k >= 0 else ":"))
        if k >= 0:
            ax.text(x, y - 3, f"{ids[k][0]}{ids[k][1]}", color="white", fontsize=5.5,
                    bbox=dict(fc=c, ec="none", pad=0.6))
    ax.axis("off")


def court(ax):
    W, H, by, r, side = 1505, 1105, 157.5, 675, 90
    ax.add_patch(plt.Rectangle((0, 0), W, H, fill=True, fc="#e8eef7", ec="k", lw=0.8))
    a = np.linspace(np.arccos((W - side - W / 2) / r), np.arccos((side - W / 2) / r), 64)
    ax.plot(W / 2 + r * np.cos(a), by + r * np.sin(a), "k", lw=0.6)
    for x in (side, W - side):
        ax.plot([x, x], [0, by + np.sqrt(r ** 2 - (x - W / 2) ** 2)], "k", lw=0.6)
    ax.add_patch(plt.Rectangle((W / 2 - 245, 0), 490, 580, fill=False, ec="k", lw=0.6))
    ax.plot(W / 2, by, "o", ms=3, color="#c0392b")
    ax.set_xlim(W + 60, -60)  # court_x grows leftwards, as seen from the cameras
    ax.set_ylim(H + 60, -60)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_views():
    """Training (held-out val, Challenge-style re-encode) vs Challenge view with our identities."""
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw=dict(width_ratios=[1280 / 470, 1180 / 330]))
    f = 3000
    fr, boxes, owner, ids, colour = identities("train720", "enc_r3_last", "IMG_0108")
    draw(axs[0], read_frame(video_path("train720", "IMG_0108"), f), fr, boxes, owner, ids, colour, f)
    axs[0].set_xlim(0, 1280); axs[0].set_ylim(620, 150)
    axs[0].set_title("Training view (IMG_0108, val)", fontsize=8)
    f = 7400
    fr, boxes, owner, ids, colour = identities("challenge", "final", "IMG_0115")
    draw(axs[1], read_frame(video_path("challenge", "IMG_0115"), f), fr, boxes, owner, ids, colour, f)
    axs[1].set_xlim(100, 1280); axs[1].set_ylim(470, 140)
    axs[1].set_title("Challenge view (IMG_0115)", fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{OUT}/views.jpg", dpi=300, bbox_inches="tight", pad_inches=0.02)


def fig_court():
    """Five seconds of predicted Challenge trajectories on the court."""
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, video, (f0, f1) in zip(axs, ("IMG_0114", "IMG_0115"), ((6000, 6150), (7400, 7550))):
        court(ax)
        P = load_frames(f"predictions/{video}.txt")
        rost = idcls.load_team_rosters("data/challenge/games_roster.csv")[video]
        colour = {team: idcls.colour_of(nums) for team, nums in rost.items()}
        frames = range(f0, f1)
        labels = P[f0][1]
        for j, (team, jersey) in enumerate(labels):
            xy = np.array([P[f][0][j] for f in frames])
            c = COLOURS[colour[team]]
            ax.plot(xy[:, 0], xy[:, 1], "-", color=c, lw=1.0, alpha=0.9)
            ax.plot(*xy[-1], "o", color=c, ms=6, mec="white", mew=0.6)
            ax.text(*xy[-1], f"{jersey}", color="white", fontsize=5, ha="center", va="center")
        ax.set_title(f"{video}, frames {f0}-{f1} (5 s)", fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{OUT}/court.pdf", bbox_inches="tight", pad_inches=0.02)


def fig_jump():
    """0114, B9 jump shot: feet leave the floor, the projected position jumps away; the stretch is
    dropped by the segment filter and interpolated."""
    video, split, tag = "IMG_0114", "challenge", "final"
    fr, tid, boxes, _ = pipeline.load_tracks(f"out/tracks/{tag}/{video}.txt")
    rost = idcls.load_team_rosters(f"data/{split}/games_roster.csv")[video]
    p, ids = pipeline.roster_probs(np.load(f"out/tracks/{tag}/{video}.probs.npy"), rost)
    owner = pipeline.assign(fr, pipeline.smooth_along_tracklets(fr, tid, p), len(ids))
    k = ids.index(("B", 9))
    m = np.where(owner == k)[0]
    m = m[np.argsort(fr[m])]
    H = load_homography(f"data/{split}/court_registration.csv")
    xy = pipeline.court_positions(H, boxes[m])
    keep = pipeline.despike(xy)
    seg = np.ones(len(m), bool)
    seg[keep] = pipeline.drop_short_segments(fr[m][keep], xy[keep])
    win = (fr[m] >= 1150) & (fr[m] <= 1210)

    fig = plt.figure(figsize=(7.0, 2.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 2.6])
    cap = cv2.VideoCapture(video_path(split, video))
    want = [int(fr[m][np.abs(fr[m] - f).argmin()]) for f in (1166, 1174, 1180, 1186)]  # nearest assigned frames
    n = 0
    shots = {}
    while len(shots) < len(want):
        ok, img = cap.read()
        n += 1
        if n in want:
            shots[n] = img
    for j, f in enumerate(want):
        i = m[fr[m] == f][0]
        x, y, w, h = boxes[i].astype(int)
        x0, y0 = max(0, x - 25), max(0, y - 25)
        crop = shots[f][y0:y + h + 15, x0:x + w + 25]
        crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(crop[..., ::-1])
        ax.add_patch(plt.Rectangle((4 * (x - x0), 4 * (y - y0)), 4 * w, 4 * h, fill=False, ec="#2e9e44", lw=0.8))
        ax.set_title(f"frame {f}", fontsize=7)
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 4])
    f = fr[m][win]
    ax.plot(f, xy[win, 1], ".", ms=2.5, color="#999999", label="raw projection")
    ok = win & keep & seg
    ax.plot(fr[m][ok], xy[ok, 1], ".", ms=2.5, color="#2e9e44", label="kept")
    pred = load_frames(f"predictions/{video}.txt")
    ff = np.arange(1150, 1211)
    j = pred[1150][1].index(("B", 9))
    ax.plot(ff, [pred[q][0][j][1] for q in ff], "-", lw=0.8, color="k", label="output")
    ax.set_xlabel("frame")
    ax.set_ylabel("court y (cm)")
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{OUT}/jump.pdf", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_views()
    fig_court()
    fig_jump()
    print("figures ->", OUT)
