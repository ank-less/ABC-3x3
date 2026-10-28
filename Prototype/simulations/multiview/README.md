# Multi-view 2D stress simulation

This experiment builds and runs eight virtual-camera clips from the first 600 frames of the held-out
Training video `IMG_0108` (about 20 seconds). It uses only Training video, Training court calibration,
Training roster, and Training court-position labels. The submitted code and weights under
`Prototype/code` remain untouched; the runner uses an isolated copy of the tools and links to the
submitted weights.

## Run

The isolated Python environment is in `.venv/`. To regenerate the base SDR clip and views, then run
the detector, tracker, identity pipeline, format check, and TI-HOTA evaluator:

```bash
./run.sh
```

The final predictions and raw tracker outputs are under `project/out/`; camera parameters and
per-view calibrations are under `project/metadata/`. The score table is `results.csv`.

## Simulated views

All clips are 1280x720, 600 frames, and use a projective warp around the court centre. The columns
below are virtual camera parameters in degrees; they are stress profiles, not measured physical
camera poses.

| Profile | Yaw | Pitch | Roll | TI-HOTA |
|---|---:|---:|---:|---:|
| base | 0° | 0° | 0° | 0.9732 |
| yaw_left_10 | -10° | 0° | 0° | 0.9739 |
| yaw_right_10 | +10° | 0° | 0° | 0.9802 |
| high_left_18 | -18° | +8° | -2° | 0.9357 |
| low_right_18 | +18° | -8° | +2° | 0.9753 |
| hard_diagonal_left | -27° | +13° | -4° | 0.8511 |
| hard_diagonal_right | +27° | -13° | +4° | 0.9365 |
| extreme_oblique | +34° | +18° | +6° | 0.7487 |

Each generated frame has a matching transformed court homography. The court-coordinate labels and
player identities remain those of the corresponding Training frame. The pipeline emitted exactly
six roster identities in all 600 frames of every profile; all eight output files passed the format
check (28,800 prediction rows total).

## Interpretation

This is a **2D perspective stress test**, not a physically rendered multi-camera dataset. A planar
homography is exact for court points; applying it to the whole image also projectively deforms the
players, who are 3D. The hard-profile scores therefore show how the current pipeline responds to
strong image-plane perspective changes. They are not estimates of TI-HOTA from a real camera at those
poses. A future physical-view simulation would need player segmentation plus a 2.5D composition, or
3D player assets and a renderer.

The baseline is a 600-frame segment, so its 0.9732 score should not be compared directly with the
paper's full-video validation score of 0.9694.
