#!/usr/bin/env python3
"""Preflight and plan the Codex + Blender realistic-human video workflow.

This script does not call a model or render frames. Codex agents use the role prompts;
Blender is the deterministic render backend once its assets are supplied.
"""
import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
TRAINING_ROOT = WORKSPACE / "Data" / "Training"


def load_json(path):
    return json.loads(Path(path).read_text())


def copy_template(src, dst):
    if not dst.exists():
        shutil.copyfile(src, dst)
        print(f"Initialized {dst.name}")


def init_files():
    copy_template(HERE / "assets.template.json", HERE / "assets.json")
    copy_template(HERE / "scene_spec.template.json", HERE / "scene_spec.json")


def find_training_video(video_id):
    return sorted(TRAINING_ROOT.rglob(f"{video_id}.MOV"))


def find_training_text(video_id, annotation_type):
    pattern = f"full player annotations/{annotation_type}/{video_id}.txt"
    return sorted(TRAINING_ROOT.rglob(pattern))


def manifest_value_path(manifest, *keys):
    value = manifest
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else HERE / path


def preflight(workflow, assets, scene):
    issues = []
    source = scene.get("source", {})
    video_id = source.get("video_id")
    purpose = source.get("purpose", "synthetic_training")
    allowed = workflow["data_policy"]["allowed_training_video_ids"]
    validation = workflow["data_policy"]["real_validation_video_ids"]
    forbidden = workflow["data_policy"]["forbidden_challenge_video_ids"]

    if video_id in forbidden:
        issues.append({"kind": "policy", "item": video_id,
                       "message": "Challenge footage is prohibited as a source for simulation."})
    elif purpose == "synthetic_training" and video_id not in allowed:
        issues.append({"kind": "policy", "item": video_id,
                       "message": "Synthetic training must use one of the six Training video IDs."})
    elif purpose == "heldout_stress_test" and video_id not in validation:
        issues.append({"kind": "policy", "item": video_id,
                       "message": "Held-out stress tests must use IMG_0108 or IMG_0109."})
    elif purpose not in ("synthetic_training", "heldout_stress_test"):
        issues.append({"kind": "policy", "item": str(purpose), "message": "Unknown scene purpose."})

    videos = find_training_video(video_id) if video_id else []
    if not videos:
        issues.append({"kind": "data", "item": str(video_id),
                       "message": "Training source video was not found under Data/Training."})
    roster = WORKSPACE / "Data" / "Training" / "Training" / "games_roster.csv"
    if not roster.exists():
        issues.append({"kind": "data", "item": str(roster), "message": "Training roster CSV is missing."})
    elif video_id:
        rows = list(csv.DictReader(roster.open(newline="")))
        if not any(r.get("video") == video_id for r in rows):
            issues.append({"kind": "data", "item": str(video_id), "message": "Video is absent from the Training roster."})
    if not list(TRAINING_ROOT.rglob("court_registration.csv")):
        issues.append({"kind": "data", "item": "court_registration.csv", "message": "Training court registration is missing."})
    if video_id and not find_training_text(video_id, "court position"):
        issues.append({"kind": "data", "item": f"{video_id} court position",
                       "message": "Training court-position annotation is missing."})

    executable = assets.get("renderer", {}).get("executable") or shutil.which("blender")
    if not executable:
        issues.append({"kind": "renderer", "item": "Blender", "message": "Blender is not installed or configured."})
    elif not Path(executable).expanduser().exists():
        issues.append({"kind": "renderer", "item": str(executable), "message": "Configured Blender executable does not exist."})

    required_assets = [
        ("court scene", ("renderer", "blend_scene")),
        ("3D court", ("court", "asset")),
        ("rigged player library", ("players", "asset_library")),
        ("front/back jersey textures", ("players", "jersey_textures_by_identity")),
        ("basketball motion library", ("motions", "asset_library")),
        ("basketball prop", ("props", "basketball_asset")),
        ("hoop prop", ("props", "hoop_asset")),
    ]
    for label, keys in required_assets:
        path = manifest_value_path(assets, *keys)
        if path is None or not path.exists():
            issues.append({"kind": "asset", "item": label,
                           "message": "Path is unset or the file does not exist."})
    if not assets.get("provenance", {}).get("licenses_checked", False):
        issues.append({"kind": "rights", "item": "asset licenses",
                       "message": "Record sources and confirm usage/redistribution rights."})

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if not issues else "blocked_on_prerequisites",
        "agent_model": workflow["agent_model"],
        "source_video_id": video_id,
        "source_purpose": purpose,
        "source_video_paths": [str(p) for p in videos],
        "frame_count": source.get("frame_count"),
        "issues": issues,
    }


def write_agent_plan(workflow):
    lines = ["# Agent execution plan", "", f"Model: `{workflow['agent_model']['id']}` (reasoning `{workflow['agent_model']['reasoning_effort']}`).", ""]
    lines.append("Use the agents in this order. Each agent returns the structured artifact named in `workflow.json`; deterministic workers validate or render those artifacts.")
    lines.append("")
    for i, agent in enumerate(workflow["agents"], 1):
        lines.append(f"{i}. **{agent['id']}** — prompt `{agent['prompt']}` → `{agent['output']}`")
    lines += ["", "## Deterministic stages", ""]
    for worker in workflow["deterministic_workers"]:
        lines.append(f"- **{worker['id']}**: {worker.get('command') or ', '.join(worker.get('outputs', []))}")
    lines += ["", "## State machine", "", " → ".join(workflow["state_machine"]), ""]
    path = HERE / "reports" / "agent_plan.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", action="store_true", help="copy templates to editable assets.json and scene_spec.json if absent")
    ap.add_argument("--plan", action="store_true", help="write the ordered agent plan")
    ap.add_argument("--preflight", action="store_true", help="check Training inputs, Blender, and 3D assets")
    args = ap.parse_args()

    if args.init:
        init_files()
    workflow = load_json(HERE / "workflow.json")
    assets_path = HERE / "assets.json"
    scene_path = HERE / "scene_spec.json"
    assets = load_json(assets_path if assets_path.exists() else HERE / "assets.template.json")
    scene = load_json(scene_path if scene_path.exists() else HERE / "scene_spec.template.json")

    if args.plan:
        print(f"Wrote {write_agent_plan(workflow)}")
    if args.preflight:
        report = preflight(workflow, assets, scene)
        report_path = HERE / "reports" / "preflight.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        print(f"Saved {report_path}")
        if report["issues"]:
            raise SystemExit(2)
    if not (args.plan or args.preflight or args.init):
        ap.print_help()


if __name__ == "__main__":
    main()
