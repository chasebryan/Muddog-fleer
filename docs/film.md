# Muddog-fleer — film design

Muddog-fleer is a standalone DOG1 visual study: a fictional FPS camera progressing through a procedural energy forest. World geometry stays fixed as the camera moves. Speed eases down for the central encounter and rises afterward; the camera never teleports.

The visual language combines dark faceted trunks, hanging energy bills/panels, emissive bark seams, ground contours, atmospheric light shafts, floating particles and a restrained thermal-style bloom. Silhouettes are faceless geometric mannequins. There are no real people, real locations, thermal data or calibrated sensor effects.

During the forest sequence the only screen-space HUD is the central reticle. Its open hyperbolic arcs contract and change color as the authored encounter progresses. The other circles and curves are beacons located in the rendered world. Opening and closing titles name the film.

## Authored sequence

| Time | Image and event |
| --- | --- |
| 0–3 s | Title fades into a continuous first-person walk. |
| 3–11 s | Camera glances toward A1, a blue silhouette; no projectiles. |
| 11–19 s | Camera eases toward T1. Blue beacons and the reticle pulse as the view centers. |
| 19.00 s | T1 changes from blue to red on a scripted game allegiance cue. |
| 20.60–21.25 s | P1 follows a curved light path from the left edge to T1. |
| 21.35–22.00 s | P2 follows a curved light path from the right edge to T1. |
| 22.10–22.75 s | P3 follows a curved light path from below to T1. |
| 23.00–25.40 s | T1 diffuses into rising light fragments and an expanding ground ring. |
| 26–39 s | Forward travel resumes; A2 remains blue, with no further projectiles. |
| 39–42 s | Closing title over the forest. |

Exactly three projectiles appear in the entire film. Their paths are screen-space Bezier animation, not a projectile physics or guidance system. Red is supplied by the scene timeline, not inferred from a silhouette's shape, location or appearance. The two A silhouettes remain blue.

## Sound

The soundtrack is generated from oscillators and seeded noise at 48 kHz in stereo. A low pump rhythm and filtered respiration texture provide an anatomical impression without recorded voices. Alternating footsteps, short beacon chirps, a state-change tone, three spatialized pulse/arrival pairs and a granular diffusion tail provide robotic activity. Beacon intervals shorten from roughly one second to three per second during the authored approach.

Signals have smooth envelopes and a master fade at both ends. The generated mix is normalized to a sample peak of 0.79 before AAC encoding. It contains no recorded weapon sounds.

## Verification and reproduction

Before rendering, the script checks that exactly three pulses exist, every launch occurs after the target is red, every arrival precedes diffusion, both A silhouettes remain blue, and the continuous camera travels forward without discontinuities at its speed boundaries. The deterministic seed is 42132.

The MP4 uses H.264, yuv420p, 24 fps, 1920 × 1080, 42 seconds and AAC stereo. `--preview` renders the scene checks and ten review frames without encoding a video. `--work-dir PATH` preserves the review images and uncompressed audio. `--output PATH` changes the output movie location and writes the corresponding JSON manifest beside it.

The geometry is projected procedurally and drawn with Pillow. It is a stylized film renderer, not a real-time FPS engine; no playable controls are included.
