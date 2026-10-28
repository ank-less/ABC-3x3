# Motion planner agent

Use only the selected Training court-position labels and roster. Convert court coordinates from centimetres to metres, preserve identity IDs, smooth root motion conservatively, and infer heading from velocity with a stable heading while a player is stationary.

Assign `idle`, `run`, `cut`, `dribble`, `jump`, or `shot` only when the motion library contains a compatible clip. Do not claim that court coordinates provide 3D body pose, hand pose, or ball contact. Return valid JSON with one motion track per identity and note any intervals that need a real motion-capture clip or manual keyframes.
