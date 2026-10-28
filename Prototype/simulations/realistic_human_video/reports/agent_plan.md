# Agent execution plan

Model: `gpt-6.1-sol` (reasoning `high`).

Use the agents in this order. Each agent returns the structured artifact named in `workflow.json`; deterministic workers validate or render those artifacts.

1. **director** — prompt `agents/director.md` → `reports/scene_plan.json`
2. **asset_curator** — prompt `agents/asset_curator.md` → `reports/asset_manifest.json`
3. **motion_planner** — prompt `agents/motion_planner.md` → `reports/motion_plan.json`
4. **camera_scene_planner** — prompt `agents/camera_scene_planner.md` → `reports/camera_plan.json`
5. **visual_qa** — prompt `agents/visual_qa.md` → `reports/qa.json`

## Deterministic stages

- **blender_renderer**: blender -b <scene.blend> --python <render_script.py>
- **annotation_exporter**: visible_masks, 2d_boxes, feet_xy_cm, identity, visibility
- **pipeline_evaluator**: run the existing detector/tracker/identity pipeline and TI-HOTA evaluator

## State machine

preflight → scene_plan → asset_manifest → motion_plan → scene_and_camera_plan → draft_render → visual_qa → repair_or_accept → final_render → synthetic_annotations → tracking_evaluation → report
