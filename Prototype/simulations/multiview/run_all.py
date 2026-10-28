#!/usr/bin/env python3
"""Run the frozen detector/tracker/identity pipeline on every generated virtual view."""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE / "project"
PYTHON = HERE / ".venv" / "bin" / "python"
TOOLS = PROJECT / "tools"
WEIGHTS = PROJECT / "weights"
REG_DIR = PROJECT / "metadata"
DATA = PROJECT / "data" / "train"
LOGS = HERE / "logs"


def run_stage(label, command, log_path, progress_path=None):
    print(f"[{label}] started", flush=True)
    started = time.time()
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        last_count = -1
        while process.poll() is None:
            time.sleep(10)
            elapsed = int(time.time() - started)
            if progress_path and progress_path.exists():
                try:
                    count = sum(1 for _ in open(progress_path))
                except OSError:
                    count = 0
                if count != last_count:
                    print(f"[{label}] {count} output rows after {elapsed}s", flush=True)
                    last_count = count
                else:
                    print(f"[{label}] running ({elapsed}s)", flush=True)
            else:
                print(f"[{label}] running ({elapsed}s)", flush=True)
        rc = process.wait()
    if rc:
        tail = "\n".join(Path(log_path).read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(f"{label} failed with exit code {rc}\n{tail}")
    print(f"[{label}] done in {int(time.time() - started)}s", flush=True)


def main():
    if not PYTHON.exists():
        raise SystemExit(f"Missing isolated Python environment: {PYTHON}")
    LOGS.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(TOOLS))
    import eval_tihota
    import check_submission
    import idcls
    from track import video_path

    rosters = idcls.load_team_rosters(str(DATA / "games_roster.csv"))

    profiles = sorted(REG_DIR.glob("SIM0108_*_court_registration.csv"))
    results = []
    for registration in profiles:
        video = registration.name.removesuffix("_court_registration.csv")
        profile = video.removeprefix("SIM0108_")
        meta = json.loads((REG_DIR / f"{video}.json").read_text())
        tag = "base" if profile == "base" else f"sim_{profile}"
        track_file = PROJECT / "out" / "tracks" / tag / f"{video}.txt"
        prediction_dir = PROJECT / "out" / "pred" / tag
        prediction_file = prediction_dir / f"{video}.txt"
        log_prefix = LOGS / f"{profile}"

        if not track_file.exists():
            run_stage(f"{profile}/track", [
                str(PYTHON), str(TOOLS / "track.py"), video,
                "--split", "train", "--weights", str(WEIGHTS / "yolo11l_r3.pt"),
                "--tag", tag, "--throttle", "0", "--device", "mps", "--imgsz", "1280",
            ], f"{log_prefix}_track.log", track_file)
        else:
            print(f"[{profile}/track] reusing {track_file}", flush=True)

        shutil.copyfile(registration, DATA / "court_registration.csv")
        if not prediction_file.exists():
            run_stage(f"{profile}/pipeline", [
                str(PYTHON), str(TOOLS / "pipeline.py"), video,
                "--split", "train", "--tag", tag,
                "--model", str(WEIGHTS / "idcls_enc.pt"), "--reclassify",
                "--out", str(prediction_dir),
            ], f"{log_prefix}_pipeline.log", prediction_file)
        else:
            print(f"[{profile}/pipeline] reusing {prediction_file}", flush=True)

        n_video_frames = check_submission.count_frames(video_path("train", video))
        format_error = check_submission.check(prediction_file, rosters[video], n_video_frames)
        if format_error:
            raise RuntimeError(f"{profile}/format failed: {format_error}")
        print(f"[{profile}/format] OK ({n_video_frames} frames x 6 identities)", flush=True)
        metrics = eval_tihota.evaluate(DATA / "court" / f"{video}.txt", prediction_file)
        result = {
            "video": video, "profile": profile, "yaw_deg": meta["yaw_deg"],
            "pitch_deg": meta["pitch_deg"], "roll_deg": meta["roll_deg"],
            "frames": metrics["frames"], "TI-HOTA": metrics["TI-HOTA"],
            "TI-DetA": metrics["TI-DetA"], "TI-AssA": metrics["TI-AssA"],
            "TI-TP": metrics["TI-TP"], "TI-FP": metrics["TI-FP"], "TI-FN": metrics["TI-FN"],
        }
        results.append(result)
        print(f"[{profile}/score] TI-HOTA={result['TI-HOTA']:.4f} "
              f"DetA={result['TI-DetA']:.4f} AssA={result['TI-AssA']:.4f} "
              f"({result['frames']} frames)", flush=True)

    results.sort(key=lambda r: r["TI-HOTA"], reverse=True)
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    with open(HERE / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("\nSummary (sorted by TI-HOTA):", flush=True)
    for r in results:
        print(f"{r['profile']:24s} HOTA={r['TI-HOTA']:.4f} "
              f"DetA={r['TI-DetA']:.4f} AssA={r['TI-AssA']:.4f}", flush=True)
    shutil.copyfile(REG_DIR / "SIM0108_base_court_registration.csv", DATA / "court_registration.csv")
    print("Restored the project calibration to the base view.", flush=True)
    print(f"Saved {HERE / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
