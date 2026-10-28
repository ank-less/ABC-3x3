# TrackID3x3 Challenge (ABC 2027): player tracking and identification in 3x3 basketball

For every frame of the Challenge videos, this code outputs the court position (cm) and the identity
(team A/B and jersey number) of each of the six players:
`frame_id,court_x,court_y,team_id,jersey_number`, one file per video, no header.

On the held-out Training videos IMG_0108 + IMG_0109, re-encoded to look like the Challenge videos,
the method scores **TI-HOTA 0.9680** (tau = 100 cm). The model was never trained on these two videos.

## Method in one picture

```
Challenge video (1280x720)
  │ 1. tools/track.py      YOLO11l player detector (fine-tuned, 3 rounds) + BoT-SORT tracker
  │ 2. tools/pipeline.py   ResNet18 identity classifier on every box: 12 (bib colour, number) + "other"
  │                        → keep the video's 6 roster identities
  │                        → average the probabilities along each tracklet (31 frames)
  │                        → per-frame Hungarian assignment of boxes to the 6 identities
  │                        → court position = homography(bottom-centre of the box)
  │                        → drop outliers (60 cm from a 15-point running median; short stretches
  │                          reachable only by > 12 m/s jumps), linear interpolation
  ▼                        → exactly 6 rows per frame, frames 1..N
predictions/IMG_0114.txt, predictions/IMG_0115.txt
```

Each team wears the same bib colour in every video (purple {4,10,12}, yellow {6,9,10}, green {6,9,11},
pink {1,3,5}), so the team and the jersey number are recognised together as one of 12 classes.

## 1. Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Weights (put them in `weights/`; they are included in the ZIP, and download links are given at the end):

| File | What | SHA-256 |
|---|---|---|
| `weights/yolo11l_r3.pt` | player detector, YOLO11l fine-tuned in 3 rounds (`tools/train_yolo.py`) | `ee9eb1a7dd2fa885672a43cdb661ce38b3eea5250cf918dac1447cf613a11e70` |
| `weights/idcls_enc.pt` | identity classifier, ResNet18 (`tools/idcls.py train --enc --strong`) | `ee15a510a42c66d23eb77edcb2976cb9ccba668571381e770cf2f48d35be3c8b` |

Public pretrained models, as allowed by the rules: Ultralytics `yolo11l.pt` (COCO) to initialise the
detector, and torchvision ResNet18 (ImageNet) to initialise the identity classifier.

## 2. Reproduce the submitted predictions (inference only)

Put the official Challenge files in `data/challenge/`:

```
data/challenge/videos/IMG_0114.MP4
data/challenge/videos/IMG_0115.MP4
data/challenge/games_roster.csv
data/challenge/court_registration.csv
```

(or run `tools/prepare_data.py --challenge "/path/to/Final challenge"`, which links them), then:

```bash
./run_challenge.sh
```

This writes `predictions/IMG_0114.txt` and `predictions/IMG_0115.txt`. It ends with
`tools/check_submission.py`, which checks the frame range against the decoded videos, the 6 roster
identities per frame and the field format. Runtime is about 25 minutes on an RTX 4050 Laptop GPU.
The output is byte-identical to the submitted files on our machine. On other GPUs, half-precision
arithmetic can change a few boxes slightly.

## 3. Validate on Training videos (IMG_0108 + IMG_0109)

```bash
.venv/bin/python tools/prepare_data.py --training "/path/to/Training"   # -> data/train (videos, boxes, GT court files)
./run_val.sh
```

`run_val.sh` re-encodes the two videos like the Challenge videos (`tools/encode720.sh`: HLG HDR to SDR
tone mapping, 1280x720, H.264 Main at 3 Mbit/s), runs the same pipeline, and scores it with
`tools/eval_tihota.py`. This scorer is equivalent to the organisers' `compute_TI-HOTA.ipynb`.
Expected output: TI-HOTA 0.9694 (IMG_0108) and 0.9665 (IMG_0109), mean 0.9680.

## 4. Re-train everything (Training split only)

Train = IMG_0104, 0105, 0106, 0107, 0110, 0111. Validation = IMG_0108 + IMG_0109, never used for training.
The Challenge videos are never used for training, pseudo-labelling or manual correction.

1. **Clean data**: `tools/prepare_data.py --training ... --challenge ...`. This step:
   - drops one off-roster box (IMG_0110, a spectator);
   - resolves 25 duplicated boxes (IMG_0105);
   - recomputes the court positions with the homography, which reproduces the official files to 0.001 cm.
2. **Detector images**: `tools/extract_frames.py` takes every 5th frame, downscaled to 1280x720, and writes
   `data/yolo720/` (14,149 train and 4,572 val images, class `player`) plus `identities/*.csv`.
3. **Challenge-style videos**: `tools/encode720.sh IMG_0104 ... IMG_0111` writes `data/train720/videos/*.mp4`
   and the `train720` split files.
4. **Detector** (Colab, NVIDIA L4):
   ```bash
   python tools/colab_make_enc.py --videos /content/train720 --data /content/yolo720   # images/{train,val}_enc
   python tools/train_yolo.py half --data /content/yolo720
   python tools/train_yolo.py r1   --data /content/yolo720 --project runs   # COCO yolo11l.pt, AdamW, early stop, best = epoch 2
   python tools/train_yolo.py r2   --data /content/yolo720 --project runs   # + all frames, SGD, 8 epochs
   python tools/train_yolo.py r3   --data /content/yolo720 --project runs   # + Challenge-style frames, rotation/shear/perspective, 6 epochs
   cp runs/ft_yolo11l_1280_r3_enc/weights/last.pt weights/yolo11l_r3.pt
   ```
   The exact arguments of each round are in `tools/train_yolo.py`. They are also stored in each checkpoint's `train_args`.
5. **Identity classifier** (local GPU):
   ```bash
   .venv/bin/python tools/idcls.py negatives      # COCO person boxes away from every player = "other"
   .venv/bin/python tools/idcls.py cache          # crops from data/yolo720
   .venv/bin/python tools/idcls.py cache-video    # the same crops from data/train720 (Challenge-style)
   .venv/bin/python tools/idcls.py train --enc --strong --out model_enc.pt
   cp data/idcls/model_enc.pt weights/idcls_enc.pt
   ```

Model selection used TI-HOTA on IMG_0108 + IMG_0109 only. The Challenge videos were used only for
label-free sanity checks (`tools/health.py`, drawn frames); their annotations were never seen.

## Files

| Path | Role |
|---|---|
| `run_challenge.sh`, `run_val.sh` | one-command inference / validation |
| `tools/track.py` | stage 1: YOLO + BoT-SORT at 1280x720 → `out/tracks/<tag>/` |
| `tools/botsort_3x3.yaml` | tracker settings (no camera-motion compensation: fixed camera) |
| `tools/pipeline.py` | stage 2: identity, assignment, homography, cleaning, interpolation → predictions |
| `tools/idcls.py` | identity classifier: data, augmentation, training, inference |
| `tools/eval_tihota.py`, `tools/diagnose.py` | TI-HOTA scorer; error breakdown by distance |
| `tools/check_submission.py` | submission format check |
| `tools/health.py` | label-free statistics (tracklets, uncertain crops, impossible jumps) |
| `tools/prepare_data.py`, `tools/extract_frames.py`, `tools/encode720.sh`, `tools/colab_make_enc.py` | data preparation |
| `tools/train_yolo.py` | detector training rounds |
| `tools/visualize.py` | drawing boxes and court maps; homography loader |
| `predictions/` | the submitted prediction files |

## Weight download links

If the ZIP upload limit forces us to leave the weights out, they are kept at fixed links until the checks end:
`weights/yolo11l_r3.pt`: <LINK-TO-ADD>, `weights/idcls_enc.pt`: <LINK-TO-ADD>
