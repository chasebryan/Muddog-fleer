# Muddog-fleer

A 42-second, first-person science-fiction film through a dark forest of faceted trunks, suspended energy panels, thermal-style light, and synthetic silhouettes. A single hyperbolic reticle accompanies the continuous walk.

**[Watch / download Muddog-fleer.mp4](media/Muddog-fleer.mp4)** · [Scene and sound design](docs/film.md)

![Muddog-fleer — three-pulse sequence](media/Muddog-fleer.jpg)

The central silhouette starts blue. A scripted allegiance cue turns it red at 19 seconds; exactly three animated light projectiles follow, then the silhouette diffuses into particles. Two other silhouettes remain blue throughout. Stereo beacon chirps accelerate during the approach, alongside synthetic respiration, pump rhythms, footsteps and pulse sounds.

This is a procedural game film. Its colors, geometry and timing are authored animation; it does not perform thermal sensing, image recognition or real-world threat classification.

## Reproduce

Requires Python 3.11+, NumPy, Pillow and FFmpeg with H.264/AAC encoding. There are no downloaded scene assets or network calls during rendering.

```sh
python -m pip install -r requirements.txt
python scripts/render_muddog_fleer.py --preview --work-dir ./preview
python scripts/render_muddog_fleer.py
```

The preview command validates camera continuity and the blue/red/projectile sequence, then produces a contact sheet. The full render saves a 1920 × 1080, 24 fps MP4 with stereo audio and a [manifest](media/Muddog-fleer.json) containing the timeline, sound cues, verification results and video checksum.

Edit [scene.json](src/muddog/scene.json) for the authored scene, or [render_muddog_fleer.py](scripts/render_muddog_fleer.py) for geometry, motion, reticle and sound design. Rendering time depends on the CPU.

## License

The repository's original [GNU Affero General Public License v3.0](LICENSE) is preserved.
