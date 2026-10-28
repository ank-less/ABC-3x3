# Agentic workflow: realistic 3x3 basketball video

This folder contains a Codex-oriented workflow specification for producing short clips with realistic 3D players, multiple virtual cameras, and machine-generated labels. It keeps the prior 2D homography stress test separate.

## Model and renderer

The workflow pins its planning and visual-QA agents to `gpt-6.1-sol` at high reasoning effort. The model plans scenes, motion, camera settings, and repairs; it does not render video pixels. GPT-6.1 Sol is a reasoning model with text output, while Blender is the configured pixel renderer. See the [official GPT-6.1 Sol model page](https://developers.openai.com/api/docs/models/gpt-6.1-sol) and [Agents API configuration guide](https://developers.openai.com/api/docs/guides/agents-api/configuration).

The active chat model cannot be changed by this folder. Use the model ID from `workflow.json` when dispatching the Codex agent stages. One architecture subtask for this workflow was run with `gpt-6.1-sol`.

## Workflow

1. **Director agent:** selects a Training-only 10-second play and outputs a structured scene plan.
2. **Asset curator:** checks six rigged human avatars, jersey textures with front/back numbers, court/hoop/ball assets, animation clips, and asset rights.
3. **Motion planner:** uses Training court-coordinate labels to drive each player's root motion, then assigns compatible actions from the motion library.
4. **Camera/scene planner:** defines calibrated physical camera poses, lighting, and ordinary plus difficult views.
5. **Blender worker:** renders a short draft, then the accepted final clip.
6. **Visual-QA agent:** checks anatomy, foot contact, action continuity, jersey readability, ball contact, occlusion, shadows, and framing; allows two repair passes.
7. **Annotation worker:** exports masks, boxes, identity, visible fraction, camera parameters, and court foot coordinates from the scene state.
8. **Evaluator:** runs the existing tracking pipeline and TI-HOTA scorer. Report visual realism and tracking metrics separately.

The Codex agents return bounded JSON artifacts. Deterministic scripts own rendering, label export, format validation, and scoring.

## Start and preflight

```bash
python3 run_workflow.py --init --plan --preflight
```

Edit `assets.json` and `scene_spec.json` after initialization. The workflow rejects Challenge video IDs. Synthetic-training clips may use IMG_0104–IMG_0107 and IMG_0110–IMG_0111; IMG_0108/IMG_0109 are reserved for held-out stress checks.

`run_workflow.py` creates `reports/agent_plan.md` and `reports/preflight.json`. It is a preflight/plan runner; it does not dispatch model calls or render. In Codex, dispatch each prompt under `agents/` using the configured model. The state machine and bounded repair policy are in `workflow.json`.

## Required inputs

- Blender executable and a scene file containing the court and calibrated cameras.
- Six rigged human avatars with realistic clothing and readable jersey numbers on the front and back.
- Basketball, hoop, and compatible idle/run/cut/dribble/jump/shot motion clips.
- Training source video and its court positions/roster.
- Asset provenance and licenses recorded in `assets.json`.

Current preflight is blocked: Blender and the 3D assets are absent, and the configured IMG_0104 source video was not found under `Data/Training`. The current workspace has no dedicated video-generation connector. The preflight report lists the missing items without substituting mannequins or calling a 2D warp photorealistic.
