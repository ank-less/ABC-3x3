# Robustness changes

- Identity probabilities use a centered 31-frame window and reset after observation gaps longer than three frames.
- Projection uncertainty uses the local homography Jacobian; despiking uses elapsed frame indices and a capped uncertainty-aware threshold.
- Short impossible stretches are measured in elapsed frames. Gaps up to 30 frames are interpolated; longer gaps use the nearest measured endpoint.
- Shrunk frames and boxes use the realized x/y resize scales; source width and height are handled independently.
- BoT-SORT retains a lost track for 30 frames rather than 90.
- The bottom-center of each box remains the court point because that matches the Training labels; the homography does not recover true foot position during jumps or occlusion.

## Validation status

These changes have not been rescored on the held-out validation videos. TI-HOTA 0.9680 in the README is the baseline from the previous pipeline version.
