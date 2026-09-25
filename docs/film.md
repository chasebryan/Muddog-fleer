# Muddog-fleer — film design

Muddog-fleer is a standalone DOG1 visual study: a fictional FPS camera progressing through a procedural energy forest. World geometry stays fixed as the camera moves. Speed eases down for the central encounter and rises afterward; the camera never teleports.

Version 2 uses shaded trunks and branches, bark furrows, roots, rocks, fallen wood and layered fern fronds. A continuous height field supplies the uneven forest floor. The monochrome environment has distance haze and restrained thermal-style bloom. Synthetic figures have shaded protective shells, vests and muted blue/red classification colors. There are no real people, real locations, thermal data or calibrated sensor effects.

Every frame is wordless, including the opening and ending. There are no title cards, captions, labels, numbers, logos or subtitle tracks. The only HUD is a small central reticle; its open hyperbolic arcs contract and change color as the authored encounter progresses. A faint trace at the synthetic marker's feet provides a world-space beacon. Large orbiting rings have been removed.

## Authored sequence

| Time | Image and event |
| --- | --- |
| 0–3 s | Forest view and reticle from the first frame; continuous first-person walk. |
| 3–11 s | Camera glances toward A1, a blue silhouette; no projectiles. |
| 11–19 s | Camera eases toward T1. Subtle blue traces and short audio beacons accompany the view centering. |
| 19.00 s | T1 changes from blue to red on a scripted game allegiance cue. |
| 20.60–21.25 s | P1 follows a short light path from below and slightly left to T1. |
| 21.35–22.00 s | P2 follows a short light path from below and slightly right to T1. |
| 22.10–22.75 s | P3 follows a short light path from below to T1. |
| 23.00–25.40 s | T1 fades into small, rising light fragments. |
| 26–39 s | Forward travel resumes; A2 remains blue, with no further projectiles. |
| 39–42 s | Forest walk continues through the final frame, with no closing title. |

Exactly three projectiles appear in the entire film. Their paths are screen-space Bezier animation, not a projectile physics or guidance system. Red is supplied by the scene timeline, not inferred from a silhouette's shape, location or appearance. The two A silhouettes remain blue.

## Sound

The soundtrack is generated from oscillators and seeded noise at 48 kHz in stereo. Wind and filtered respiration supply the ambient texture, with a subdued pump rhythm. Alternating footfalls, quiet beacon chirps, short mechanical clicks and three spatialized pulse/arrival pairs provide restrained robotic activity. Beacon intervals shorten from roughly one second to three per second during the authored approach. No spoken words are present.

Signals have smooth envelopes and a master fade at both ends. The generated mix is normalized to a sample peak of 0.79 before AAC encoding. It contains no recorded weapon sounds.

## Verification and reproduction

Before rendering, the script checks that exactly three pulses exist, every launch occurs after the target is red, every arrival precedes diffusion, both A silhouettes remain blue, and the continuous camera travels forward without discontinuities at its speed boundaries. An AST check rejects text-drawing calls anywhere in the renderer. No fonts or text assets are loaded. The deterministic seed is 42132.

The MP4 uses H.264, yuv420p, 24 fps, 1920 × 1080, 42 seconds and AAC stereo. CRF 24 with a 1,900 kbit/s maximum rate keeps the repository download compact. `--preview` renders the scene checks and ten wordless review frames, including the first and last frame, without encoding a video. `--work-dir PATH` preserves the review images and uncompressed audio. `--output PATH` changes the output movie location and writes the corresponding JSON manifest beside it. `--workers 1` through `--workers 4` selects the number of parallel frame workers; projection of static scenery is batched.

The geometry is projected procedurally and drawn with Pillow. It is a stylized film renderer, not a real-time FPS engine; no playable controls are included.
