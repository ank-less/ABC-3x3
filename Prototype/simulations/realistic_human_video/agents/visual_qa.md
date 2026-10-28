# Visual QA agent

Review the draft contact sheet and short clips. Inspect player anatomy, foot contact, foot sliding, jersey number readability, stable identity appearance, ball/hand contact, shadows, intersections, court perspective, occlusions, and frame-to-frame continuity.

Return valid JSON with `pass`, per-issue severity, affected frames/cameras, a bounded repair request, and unresolved defects. Allow at most two repair passes. Never approve solely because the detector score is high; separate visual realism, physical consistency, annotation correctness, and tracking metrics.
