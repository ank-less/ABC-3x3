"""TI-HOTA evaluator, equivalent to the organizers' compute_TI-HOTA.ipynb
(https://github.com/open-starlab/TrackID3x3/blob/main/script/compute_TI-HOTA.ipynb).

    .venv/bin/python tools/eval_tihota.py --pred out/pred                  # every GT video with a prediction
    .venv/bin/python tools/eval_tihota.py --pred out/pred --videos IMG_0108 IMG_0109
    .venv/bin/python tools/eval_tihota.py --selftest

Both GT and prediction files use the submission format, no header:
    frame_id,court_x,court_y,team_id,jersey_number
The identity attribute is (team_id, jersey_number). Same rules as the official code:
per frame Hungarian matching on cost 1 - LocSim * IdSim with pairs below alpha excluded,
DetA = TP / (TP + FP + FN), AssA = unweighted mean over GT identities of TPA / (TPA + FNA + FPA),
TI-HOTA = mean over alpha in 0.05..0.95 of sqrt(DetA * AssA). The only change is speed:
the similarity matrix of a frame is computed once and reused for all 19 alphas.
"""
import argparse
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

TAU = 100.0
ALPHAS = np.arange(0.05, 1.0, 0.05)
ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_frames(path):
    """Return {frame: (positions (n, 2), [identity, ...])} from a submission-format file."""
    rows = defaultdict(list)
    for line in open(path):
        p = line.strip().split(",")
        if len(p) < 5:
            continue
        rows[int(p[0])].append((float(p[1]), float(p[2]), (p[3].strip(), int(float(p[4])))))
    return {f: (np.array([[x, y] for x, y, _ in r], float).reshape(-1, 2), [i for _, _, i in r])
            for f, r in rows.items()}


def evaluate(gt_path, pred_path, tau=TAU):
    gt, pred = load_frames(gt_path), load_frames(pred_path)
    empty = (np.zeros((0, 2)), [])
    n_alpha = len(ALPHAS)
    tp = np.zeros(n_alpha)
    fp = np.zeros(n_alpha)
    fn = np.zeros(n_alpha)
    tpa = defaultdict(lambda: np.zeros(n_alpha))
    fna = defaultdict(lambda: np.zeros(n_alpha))
    fpa = defaultdict(lambda: np.zeros(n_alpha))
    frames = sorted(set(gt) | set(pred))
    for f in frames:
        g_pos, g_id = gt.get(f, empty)
        p_pos, p_id = pred.get(f, empty)
        d2 = ((g_pos[:, None, :] - p_pos[None, :, :]) ** 2).sum(-1)
        same = np.array([[gi == pi for pi in p_id] for gi in g_id], float).reshape(len(g_id), len(p_id))
        combined = np.exp(math.log(0.05) * d2 / tau ** 2) * same
        for gid in g_id:
            tpa[gid]  # every GT identity counts in AssA even if never matched
        for a, alpha in enumerate(ALPHAS):
            cost = 1 - combined
            cost[combined < alpha] = 1e6
            rows, cols = linear_sum_assignment(cost)
            keep = combined[rows, cols] >= alpha
            m_gt, m_pred = set(rows[keep]), set(cols[keep])
            n_tp = len(m_gt)
            tp[a] += n_tp
            fp[a] += len(p_id) - n_tp
            fn[a] += len(g_id) - n_tp
            for i, gid in enumerate(g_id):
                if i in m_gt:
                    tpa[gid][a] += 1
                else:
                    fna[gid][a] += 1
            for j, pid in enumerate(p_id):
                if j not in m_pred:
                    fpa[pid][a] += 1
    det = tp / np.maximum(tp + fp + fn, 1)
    ass = np.zeros(n_alpha)
    for a in range(n_alpha):
        vals = [tpa[g][a] / d for g in tpa if (d := tpa[g][a] + fna[g][a] + fpa[g][a]) > 0]
        ass[a] = np.mean(vals) if vals else 0.0
    hota = np.sqrt(det * ass)
    return {
        "TI-HOTA": float(hota.mean()), "TI-DetA": float(det.mean()), "TI-AssA": float(ass.mean()),
        "TI-TP": float(tp.mean()), "TI-FP": float(fp.mean()), "TI-FN": float(fn.mean()),
        "frames": len(frames), "TI-HOTA_alphas": hota.tolist(),
    }


def report(gt_dir, pred_dir, videos=None):
    names = videos or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(pred_dir, "*.txt")))
    results = {}
    for name in names:
        gt_path, pred_path = os.path.join(gt_dir, f"{name}.txt"), os.path.join(pred_dir, f"{name}.txt")
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"{name}: missing GT or prediction, skipped", file=sys.stderr)
            continue
        m = results[name] = evaluate(gt_path, pred_path)
        print(f"{name}  TI-HOTA {m['TI-HOTA']:.4f}  DetA {m['TI-DetA']:.4f}  AssA {m['TI-AssA']:.4f}  "
              f"TP {m['TI-TP']:.0f}  FP {m['TI-FP']:.0f}  FN {m['TI-FN']:.0f}  frames {m['frames']}")
    if len(results) > 1:
        for key in ("TI-HOTA", "TI-DetA", "TI-AssA"):
            vals = [m[key] for m in results.values()]
            print(f"mean {key}: {np.mean(vals):.4f} (std {np.std(vals):.4f})")
    return results


def selftest(gt_dir):
    """Score perturbed copies of one GT file to check the evaluator behaves as expected."""
    import tempfile
    src = os.path.join(gt_dir, "IMG_0109.txt")
    lines = [l.strip().split(",") for l in open(src)]
    rng = np.random.default_rng(0)
    cases = {
        "identical": lambda r: [r],
        "noise 13 cm": lambda r: [[r[0], f"{float(r[1]) + rng.normal(0, 13 / 1.414):.3f}",
                                   f"{float(r[2]) + rng.normal(0, 13 / 1.414):.3f}", r[3], r[4]]],
        "noise 50 cm": lambda r: [[r[0], f"{float(r[1]) + rng.normal(0, 50 / 1.414):.3f}",
                                   f"{float(r[2]) + rng.normal(0, 50 / 1.414):.3f}", r[3], r[4]]],
        "drop odd frames": lambda r: [r] if int(r[0]) % 2 == 0 else [],
        "wrong jersey for B": lambda r: [r[:4] + ["99"]] if r[3] == "B" else [r],
        "extra duplicate rows": lambda r: [r, r[:1] + [str(float(r[1]) + 300), r[2], r[3], r[4]]],
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in cases.items():
            path = os.path.join(tmp, "pred.txt")
            with open(path, "w") as f:
                for r in lines:
                    for out in fn(r):
                        f.write(",".join(out) + "\n")
            m = evaluate(src, path)
            print(f"{name:22s} TI-HOTA {m['TI-HOTA']:.4f}  DetA {m['TI-DetA']:.4f}  AssA {m['TI-AssA']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=os.path.join(ROOT, "data", "train", "court"))
    ap.add_argument("--pred", help="folder of IMG_xxxx.txt predictions")
    ap.add_argument("--videos", nargs="*", help="limit to these videos (default: every prediction file)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.gt)
    elif args.pred:
        report(args.gt, args.pred, args.videos)
    else:
        ap.error("give --pred or --selftest")


if __name__ == "__main__":
    main()
