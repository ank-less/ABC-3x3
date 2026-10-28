# Camera and scene planner agent

Create camera settings from the requested physical camera positions, height, look-at point, focal length, resolution, and frame rate. Plan one ordinary view and at least two difficult views, including a low angle and an occlusion-heavy angle.

Return valid JSON with Blender camera parameters, lighting, framing, and the corresponding intrinsics/extrinsics (`K`, `R`, `t`). Derive the court homography from each camera and the court plane. Ensure the six players stay inside the useful image area. Do not use a whole-frame image warp as a substitute for a 3D camera.
