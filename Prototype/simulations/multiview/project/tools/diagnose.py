"""Per-identity error breakdown of a prediction against GT (both in submission format).

    .venv/bin/python tools/diagnose.py --pred out/pred/coco_l --videos IMG_0108 IMG_0109

For each identity (team, jersey): share of frames whose predicted position is within 30 / 100 / 200 cm
of the GT position of the same identity, and the median error. Low "<200cm" means the identity was given
to the wrong person (or missing for long stretches); good "<200cm" but low "<30cm" means localisation.

Then "points lost": each (frame, identity) earns about the share of the 19 alphas its similarity passes
(TI-HOTA matches only same-identity pairs), so 1 - that share is what it loses. The loss is split by error
size: <15 cm (box precision), 15-50 cm (rough box / occlusion), 50-200 cm (near miss, often a gap
guess), >200 cm or missing (wrong person / long gap).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_tihota import ALPHAS, TAU, load_frames  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def per_identity(gt_path, pred_path):
    gt, pred = load_frames(gt_path), load_frames(pred_path)
    err = {}
    for f, (gpos, gids) in gt.items():
        ppos, pids = pred.get(f, (np.zeros((0, 2)), []))
        where = {i: k for k, i in enumerate(pids)}
        for g, ident in zip(gpos, gids):
            k = where.get(ident)
            err.setdefault(ident, []).append(np.inf if k is None else np.hypot(*(ppos[k] - g)))
    return {i: np.array(e) for i, e in sorted(err.items())}


def earned(e):
    sim = np.exp(np.log(0.05) * np.where(np.isfinite(e), e, 1e9) ** 2 / TAU ** 2)
    return (sim[:, None] >= ALPHAS[None]).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default=os.path.join(ROOT, "data", "train", "court"))
    ap.add_argument("--videos", nargs="+", required=True)
    args = ap.parse_args()
    for v in args.videos:
        print(f"{v}:  identity   <30cm  <100cm  <200cm  median cm")
        for ident, e in per_identity(os.path.join(args.gt, f"{v}.txt"), os.path.join(args.pred, f"{v}.txt")).items():
            print(f"    {ident[0]}{ident[1]:<3d}     {(e < 30).mean():5.0%}  {(e < 100).mean():6.0%}  "
                  f"{(e < 200).mean():6.0%}  {np.median(e):8.0f}")
        e = np.concatenate(list(per_identity(os.path.join(args.gt, f"{v}.txt"), os.path.join(args.pred, f"{v}.txt")).values()))
        lost = 1 - earned(e)
        parts = [("<15cm", e < 15), ("15-50cm", (e >= 15) & (e < 50)), ("50-200cm", (e >= 50) & (e < 200)), (">200cm/missing", e >= 200)]
        print(f"    points earned {1 - lost.mean():.3f}; lost " + ", ".join(
            f"{name} {lost[m].sum() / len(e):.3f} ({m.mean():.1%} of rows)" for name, m in parts))


if __name__ == "__main__":
    main()
