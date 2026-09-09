# grid-pour-mujoco

A MuJoCo reproduction of the dual-arm lab demo from [@genrobotics_ai's GRID announcement](https://x.com/genrobotics_ai/status/2097713940893679902): one arm picks a test tube of pink liquid from a rack, hands it to the second arm, which pours it into a beaker — then hands it back and re-racks it. Their demo coupled a custom NVIDIA Warp fluid solver with MuJoCo; this rebuild gets the same beat with pure MuJoCo: 130 rigid particle spheres stand in for the liquid, kinematically pinned inside the tube during carries, released on a stagger during the pour, and absorbed into a rising fill volume in the beaker by volume conservation. Final pour capture: 130/130 particles.

![demo](demo.mp4)

## What's in the scene

- Two procedural 7-DOF Flexiv-style arms with parallel grippers, driven by damped-least-squares IK tracking min-jerk Cartesian keyframes
- Runtime weld constraints for the pick, the mid-air handoff between arms, and the re-rack
- Test tube and beaker with transparent glass visuals and ring-of-boxes collision walls
- 130 pink particles (the "liquid"), a fill cylinder that rises with absorbed volume
- Light-studio look: seamless backdrop, soft key/fill/wash lights, slow camera orbit with a push-in on the pour
- 29 s, 1280x720 @ 30 fps, rendered headless (GLFW + xvfb), no GPU required

## Run it

```bash
pip install mujoco imageio numpy
python3 gen_scene.py            # writes scene.xml
xvfb-run -a python3 sim.py      # simulates + renders demo.mp4
```

`sim.py` prints the absorbed-particle count at the end (130/130 on a good run) and needs `ffmpeg` on PATH for the video encode.

## Files

- `gen_scene.py` - procedural scene generator (arms, bench, glassware, particles) -> `scene.xml`
- `sim.py` - IK, choreography, particle pinning/release/absorption, offscreen render -> `demo.mp4`
- `demo.mp4` - the rendered run

Inspired by the GRID lab demo from Generalist Robotics ([@genrobotics_ai on X](https://x.com/genrobotics_ai/status/2097713940893679902)). Unofficial fan rebuild, not affiliated.
