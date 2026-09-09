"""Generate scene.xml for the GRID-style dual-arm lab pour in MuJoCo.

Recreation of General Robotics' Auto-Engineering demo (Sep 9, 2026):
two Flexiv-style 7-DOF arms pick a test tube from a rack, pass it
between arms, and pour pink liquid into a beaker. Liquid is real
particles; the beaker fills by volume conservation.
"""
import math, random

# ---- collision bit plan ----
# statics: contype=1 conaffinity=7   (bench, floor, rack, walls, backdrop)
# particles: contype=2 conaffinity=9 (statics + container walls)
# arms: contype=4 conaffinity=1      (statics only)
# container walls (tube/beaker): contype=8 conaffinity=2 (particles only)
# pure visual: 0/0

SILVER = "0.79 0.81 0.83 1"
WHITE  = "0.95 0.95 0.96 1"
BLUE   = "0.45 0.83 0.95 1"
BLACK  = "0.08 0.08 0.09 1"
GLASS  = "0.9 0.95 1.0 0.22"
PINK   = "1.0 0.25 0.6 0.95"
ORANGE = "0.95 0.55 0.1 1"

TUBE_R   = 0.013      # test tube inner wall ring radius
TUBE_H   = 0.13       # tube height
BEAK_R   = 0.028      # beaker inner ring radius
BEAK_H   = 0.08       # beaker height
BENCH_Z  = 0.75       # bench top
P_RAD    = 0.0028     # liquid particle radius
N_PART   = 175        # liquid particles (cap)

RACK_POS  = (-0.38, -0.12)          # tube start, on bench
BEAK_POS  = ( 0.30, -0.14)          # beaker, on bench
TUBE_Z0   = BENCH_Z + 0.002         # tube bottom
TUBE_C0   = (RACK_POS[0], RACK_POS[1], TUBE_Z0 + TUBE_H/2)

def ring_boxes(prefix, R, z0, h, n, thick=0.0025, extra=0.0006, rgba="0 0 0 0"):
    """Collision-only ring of thin boxes approximating a cylinder wall."""
    out = []
    halfw = math.pi * R / n + extra
    for i in range(n):
        th = 2 * math.pi * i / n
        x, y = R * math.cos(th), R * math.sin(th)
        deg = math.degrees(th)
        out.append(
            f'<geom name="{prefix}{i}" type="box" size="{thick/2:.5f} {halfw:.5f} {h/2:.5f}" '
            f'pos="{x:.5f} {y:.5f} {z0 + h/2:.5f}" euler="0 0 {deg:.3f}" '
            f'contype="8" conaffinity="2" rgba="{rgba}" mass="0"/>')
    return "\n".join(out)

def particle_layout():
    """Deterministic non-overlapping grid packing inside the upright tube."""
    pts = []
    r_max = TUBE_R - P_RAD - 0.0018
    step = P_RAD * 2.06
    z = P_RAD + 0.0075   # above 6mm-thick bottom cap
    layer = 0
    while len(pts) < N_PART and z < TUBE_H - P_RAD - 0.004:
        off = (step / 2) if layer % 2 else 0.0
        k = int(r_max / step) + 1
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                if len(pts) >= N_PART:
                    break
                x, y = i * step + off, j * step + off
                if x * x + y * y <= r_max * r_max:
                    pts.append((x, y, z))
        z += step
        layer += 1
    return pts[:N_PART]

def arm(prefix, base_pos, yaw_deg):
    """Flexiv Rizon-style 7-DOF arm, links along local +z, pitch joints fold forward."""
    p = base_pos
    b = []
    b.append(f'''
<body name="{prefix}base" pos="{p[0]} {p[1]} {p[2]}" euler="0 0 {yaw_deg}">
  <inertial mass="2.0" pos="0 0 0.03" diaginertia="0.004 0.004 0.004"/>
  <geom name="{prefix}ped" type="cylinder" size="0.065 0.035" pos="0 0 0.035" rgba="{WHITE}" contype="4" conaffinity="1"/>
  <geom name="{prefix}ped2" type="cylinder" size="0.058 0.05" pos="0 0 0.1" rgba="{SILVER}" contype="4" conaffinity="1"/>
  <geom name="{prefix}ring0" type="cylinder" size="0.0595 0.004" pos="0 0 0.145" rgba="{BLUE}" contype="0" conaffinity="0"/>
  <joint name="{prefix}j1" type="hinge" axis="0 0 1" pos="0 0 0.15" range="-170 170" damping="6" armature="0.02"/>
  <body name="{prefix}l1" pos="0 0 0.15">
    <inertial mass="1.2" pos="0 0 0.08" diaginertia="0.004 0.004 0.001"/>
    <geom name="{prefix}sh" type="capsule" fromto="0 0 -0.01 0 0 0.15" size="0.052" rgba="{SILVER}" contype="4" conaffinity="1"/>
    <geom name="{prefix}cap1" type="sphere" size="0.052" pos="0 0 0.15" rgba="{WHITE}" contype="0" conaffinity="0"/>
    <joint name="{prefix}j2" type="hinge" axis="0 1 0" pos="0 0 0.15" range="-140 140" damping="6" armature="0.02"/>
    <body name="{prefix}l2" pos="0 0 0.15">
      <inertial mass="1.0" pos="0 0 0.13" diaginertia="0.006 0.006 0.001"/>
      <geom name="{prefix}ua" type="capsule" fromto="0 0 0.005 0 0 0.27" size="0.046" rgba="{SILVER}" contype="4" conaffinity="1"/>
      <geom name="{prefix}ring2" type="cylinder" size="0.0475 0.004" pos="0 0 0.02" rgba="{BLUE}" contype="0" conaffinity="0"/>
      <joint name="{prefix}j3" type="hinge" axis="0 0 1" pos="0 0 0.27" range="-170 170" damping="3" armature="0.01"/>
      <body name="{prefix}l3" pos="0 0 0.27">
        <inertial mass="0.8" pos="0 0 0.03" diaginertia="0.002 0.002 0.001"/>
        <geom name="{prefix}eb" type="capsule" fromto="0 0 -0.005 0 0 0.055" size="0.047" rgba="{SILVER}" contype="4" conaffinity="1"/>
        <geom name="{prefix}cap3" type="sphere" size="0.047" pos="0 0 0" rgba="{WHITE}" contype="0" conaffinity="0"/>
        <joint name="{prefix}j4" type="hinge" axis="0 1 0" pos="0 0 0.055" range="-140 140" damping="3" armature="0.01"/>
        <body name="{prefix}l4" pos="0 0 0.055">
          <inertial mass="0.7" pos="0 0 0.11" diaginertia="0.004 0.004 0.0008"/>
          <geom name="{prefix}fa" type="capsule" fromto="0 0 0.005 0 0 0.23" size="0.04" rgba="{SILVER}" contype="4" conaffinity="1"/>
          <geom name="{prefix}ring4" type="cylinder" size="0.0415 0.004" pos="0 0 0.02" rgba="{BLUE}" contype="0" conaffinity="0"/>
          <joint name="{prefix}j5" type="hinge" axis="0 0 1" pos="0 0 0.23" range="-170 170" damping="1.5" armature="0.008"/>
          <body name="{prefix}l5" pos="0 0 0.23">
            <inertial mass="0.4" pos="0 0 0.025" diaginertia="0.001 0.001 0.0005"/>
            <geom name="{prefix}wr" type="capsule" fromto="0 0 -0.004 0 0 0.05" size="0.04" rgba="{SILVER}" contype="4" conaffinity="1"/>
            <joint name="{prefix}j6" type="hinge" axis="0 1 0" pos="0 0 0.05" range="-120 120" damping="1.5" armature="0.008"/>
            <body name="{prefix}l6" pos="0 0 0.05">
              <inertial mass="0.3" pos="0 0 0.025" diaginertia="0.0008 0.0008 0.0003"/>
              <geom name="{prefix}wm" type="cylinder" size="0.036 0.028" pos="0 0 0.028" rgba="{BLACK}" contype="4" conaffinity="1"/>
              <joint name="{prefix}j7" type="hinge" axis="0 0 1" pos="0 0 0.056" range="-170 170" damping="1.0" armature="0.005"/>
              <body name="{prefix}hand" pos="0 0 0.056">
                <inertial mass="0.35" pos="0 0 0.05" diaginertia="0.0008 0.0008 0.0004"/>
                <geom name="{prefix}palm" type="box" size="0.02 0.042 0.014" pos="0 0 0.014" rgba="{BLACK}" contype="4" conaffinity="1"/>
                <geom name="{prefix}wrist_ring" type="cylinder" size="0.037 0.004" pos="0 0 0.002" rgba="{BLUE}" contype="0" conaffinity="0"/>
                <site name="{prefix}ee" pos="0 0 0.115" size="0.004" rgba="1 0 0 0"/>
                <body name="{prefix}fA" pos="0 0.038 0.028">
                  <inertial mass="0.05" pos="0 0 0.045" diaginertia="0.00005 0.00005 0.00002"/>
                  <joint name="{prefix}fAs" type="slide" axis="0 -1 0" range="0 0.026" damping="0.5"/>
                  <geom name="{prefix}fAg" type="box" size="0.011 0.011 0.05" pos="0 0 0.05" rgba="{BLACK}" contype="4" conaffinity="1"/>
                  <geom name="{prefix}fAt" type="box" size="0.008 0.013 0.012" pos="0 -0.001 0.095" rgba="{BLACK}" contype="4" conaffinity="1"/>
                </body>
                <body name="{prefix}fB" pos="0 -0.038 0.028">
                  <inertial mass="0.05" pos="0 0 0.045" diaginertia="0.00005 0.00005 0.00002"/>
                  <joint name="{prefix}fBs" type="slide" axis="0 1 0" range="0 0.026" damping="0.5"/>
                  <geom name="{prefix}fBg" type="box" size="0.011 0.011 0.05" pos="0 0 0.05" rgba="{BLACK}" contype="4" conaffinity="1"/>
                  <geom name="{prefix}fBt" type="box" size="0.008 0.013 0.012" pos="0 0.001 0.095" rgba="{BLACK}" contype="4" conaffinity="1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </body>
</body>''')
    return "\n".join(b)

def build_xml():
    parts = particle_layout()
    n_actual = len(parts)

    # particles (world children, welded to tube until the pour)
    pxml = []
    eqxml = []
    for i, (x, y, z) in enumerate(parts):
        wx, wy, wz = TUBE_C0[0] + x, TUBE_C0[1] + y, TUBE_C0[2] - TUBE_H/2 + z
        pxml.append(f'''
<body name="p{i}" pos="{wx:.5f} {wy:.5f} {wz:.5f}">
  <freejoint name="pf{i}"/>
  <inertial mass="0.00018" pos="0 0 0" diaginertia="1e-9 1e-9 1e-9"/>
  <geom name="pg{i}" type="sphere" size="{P_RAD}" rgba="{PINK}" contype="2" conaffinity="9"
        friction="0.6 0.01 0.03" solref="0.005 1" solimp="0.95 0.99 0.0005"/>
</body>''')
        # weld in tube frame: relpos = particle pos relative to tube origin (tube center)
        pass

    xml = f'''<mujoco model="grid_pour">
  <compiler angle="degree" inertiafromgeom="false"/>
  <option iterations="150" tolerance="1e-10" timestep="0.002" solver="Newton">
    <flag warmstart="enable"/>
  </option>
  <visual>
    <global offwidth="1280" offheight="720" azimuth="35" elevation="-20"/>
    <quality shadowsize="2048" offsamples="4"/>
    <map znear="0.02" zfar="10"/>
      </visual>
  <worldbody>
    <light name="key" directional="true" pos="1.2 -1.6 2.6" dir="-0.45 0.6 -1" castshadow="true" diffuse="1.1 1.1 1.1" specular="0.3 0.3 0.3" ambient="0.45 0.45 0.47"/>
    <light name="wash" directional="true" pos="0 -2.0 1.2" dir="0 1 0.15" castshadow="false" diffuse="0.8 0.8 0.82"/>
    <light name="fill" directional="true" pos="-1.5 -0.6 1.8" dir="0.7 0.25 -1" castshadow="false" diffuse="0.35 0.35 0.38"/>
    <light name="rim" directional="true" pos="0.4 1.6 2.2" dir="-0.1 -0.7 -1" castshadow="false" diffuse="0.3 0.32 0.35"/>
    <geom name="floor" type="plane" size="6 6 0.1" rgba="0.9 0.9 0.89 1" contype="1" conaffinity="7"/>
    <geom name="wall" type="plane" pos="0 0.95 1.5" zaxis="0 -1 0" size="6 6 0.1" rgba="0.94 0.94 0.93 1" contype="0" conaffinity="0"/>

    <!-- lab bench -->
    <geom name="bench" type="box" size="0.95 0.4 0.025" pos="0 0 {BENCH_Z-0.025}" rgba="0.84 0.71 0.52 1" contype="1" conaffinity="7"/>
    <geom name="bench_edge" type="box" size="0.95 0.4 0.006" pos="0 0 {BENCH_Z-0.05}" rgba="0.25 0.22 0.2 1" contype="0" conaffinity="0"/>
    <geom name="leg1" type="box" size="0.03 0.03 0.35" pos="-0.85 -0.3 0.35" rgba="0.3 0.3 0.31 1" contype="1" conaffinity="7"/>
    <geom name="leg2" type="box" size="0.03 0.03 0.35" pos="0.85 -0.3 0.35" rgba="0.3 0.3 0.31 1" contype="1" conaffinity="7"/>
    <geom name="leg3" type="box" size="0.03 0.03 0.35" pos="-0.85 0.3 0.35" rgba="0.3 0.3 0.31 1" contype="1" conaffinity="7"/>
    <geom name="leg4" type="box" size="0.03 0.03 0.35" pos="0.85 0.3 0.35" rgba="0.3 0.3 0.31 1" contype="1" conaffinity="7"/>

    <!-- orange tube rack -->
    <geom name="rack_base" type="box" size="0.055 0.055 0.006" pos="{RACK_POS[0]} {RACK_POS[1]} {BENCH_Z+0.006}" rgba="{ORANGE}" contype="1" conaffinity="7"/>
    <geom name="rack_top" type="box" size="0.055 0.055 0.005" pos="{RACK_POS[0]} {RACK_POS[1]} {BENCH_Z+0.1}" rgba="{ORANGE}" contype="1" conaffinity="7"/>
    <geom name="rack_p1" type="box" size="0.004 0.004 0.05" pos="{RACK_POS[0]-0.05} {RACK_POS[1]-0.05} {BENCH_Z+0.055}" rgba="{ORANGE}" contype="1" conaffinity="7"/>
    <geom name="rack_p2" type="box" size="0.004 0.004 0.05" pos="{RACK_POS[0]+0.05} {RACK_POS[1]-0.05} {BENCH_Z+0.055}" rgba="{ORANGE}" contype="1" conaffinity="7"/>
    <geom name="rack_p3" type="box" size="0.004 0.004 0.05" pos="{RACK_POS[0]-0.05} {RACK_POS[1]+0.05} {BENCH_Z+0.055}" rgba="{ORANGE}" contype="1" conaffinity="7"/>
    <geom name="rack_p4" type="box" size="0.004 0.004 0.05" pos="{RACK_POS[0]+0.05} {RACK_POS[1]+0.05} {BENCH_Z+0.055}" rgba="{ORANGE}" contype="1" conaffinity="7"/>

    <!-- back-shelf glassware props -->
    <geom name="bottle1" type="cylinder" size="0.035 0.09" pos="-0.62 0.28 {BENCH_Z+0.09}" rgba="0.85 0.7 0.3 0.45" contype="1" conaffinity="7"/>
    <geom name="bottle2" type="cylinder" size="0.028 0.07" pos="-0.52 0.3 {BENCH_Z+0.07}" rgba="0.6 0.85 0.9 0.4" contype="1" conaffinity="7"/>
    <geom name="flask" type="cylinder" size="0.045 0.06" pos="0.62 0.3 {BENCH_Z+0.06}" rgba="0.9 0.95 1 0.35" contype="1" conaffinity="7"/>

    <!-- test tube: free body, glass visual + collision ring + bottom -->
    <body name="tube" pos="{TUBE_C0[0]} {TUBE_C0[1]} {TUBE_C0[2]}">
      <freejoint name="tubej"/>
      <inertial mass="0.02" pos="0 0 0" diaginertia="3e-5 3e-5 1e-5"/>
      <geom name="tube_glass" type="capsule" fromto="0 0 {-TUBE_H/2} 0 0 {TUBE_H/2 - 0.013}" size="{TUBE_R}" rgba="{GLASS}" contype="0" conaffinity="0"/>
      <geom name="tube_bottom" type="cylinder" size="{TUBE_R-0.0002} 0.003" pos="0 0 {-TUBE_H/2 + 0.003}" contype="8" conaffinity="2" rgba="0 0 0 0" mass="0"/>
{ring_boxes('tw', TUBE_R, -TUBE_H/2, TUBE_H, 12)}
    </body>

    <!-- beaker -->
    <body name="beaker" pos="{BEAK_POS[0]} {BEAK_POS[1]} {BENCH_Z}">
      <inertial mass="0.4" pos="0 0 0.03" diaginertia="0.001 0.001 0.001"/>
      <geom name="beak_glass" type="cylinder" size="{BEAK_R} {BEAK_H/2}" pos="0 0 {BEAK_H/2}" rgba="{GLASS}" contype="0" conaffinity="0"/>
      <geom name="beak_bottom" type="cylinder" size="{BEAK_R-0.0002} 0.004" pos="0 0 0.004" contype="8" conaffinity="2" rgba="0 0 0 0" mass="0"/>
      <geom name="beak_liquid" type="cylinder" size="{BEAK_R-0.0022} 0.0001" pos="0 0 0.003" rgba="1.0 0.25 0.6 0.85" contype="0" conaffinity="0"/>
{ring_boxes('bw', BEAK_R, 0, BEAK_H, 16)}
    </body>

{arm('L', (-0.52, 0.06, BENCH_Z), 0)}
{arm('R', (0.52, 0.06, BENCH_Z), 180)}

{''.join(pxml)}
  </worldbody>

  <equality>
    <weld name="world_tube" body1="world" body2="tube" anchor="0 0 0" active="true" torquescale="100" solref="0.005 1"/>
    <weld name="L_tube" body1="Lhand" body2="tube" anchor="0 0 0" active="false" torquescale="100" solref="0.005 1"/>
    <weld name="R_tube" body1="Rhand" body2="tube" anchor="0 0 0" active="false" torquescale="100" solref="0.005 1"/>
{''.join(eqxml)}
  </equality>

  <actuator>
    <!-- left arm -->
    <position name="aL1" joint="Lj1" kp="400" ctrlrange="-170 170" forcerange="-40 40"/>
    <position name="aL2" joint="Lj2" kp="400" ctrlrange="-140 140" forcerange="-40 40"/>
    <position name="aL3" joint="Lj3" kp="200" ctrlrange="-170 170" forcerange="-20 20"/>
    <position name="aL4" joint="Lj4" kp="300" ctrlrange="-140 140" forcerange="-25 25"/>
    <position name="aL5" joint="Lj5" kp="120" ctrlrange="-170 170" forcerange="-10 10"/>
    <position name="aL6" joint="Lj6" kp="150" ctrlrange="-120 120" forcerange="-10 10"/>
    <position name="aL7" joint="Lj7" kp="80"  ctrlrange="-170 170" forcerange="-6 6"/>
    <position name="aLfA" joint="LfAs" kp="60" ctrlrange="0 0.026" forcerange="-3 3"/>
    <position name="aLfB" joint="LfBs" kp="60" ctrlrange="0 0.026" forcerange="-3 3"/>
    <!-- right arm -->
    <position name="aR1" joint="Rj1" kp="400" ctrlrange="-170 170" forcerange="-40 40"/>
    <position name="aR2" joint="Rj2" kp="400" ctrlrange="-140 140" forcerange="-40 40"/>
    <position name="aR3" joint="Rj3" kp="200" ctrlrange="-170 170" forcerange="-20 20"/>
    <position name="aR4" joint="Rj4" kp="300" ctrlrange="-140 140" forcerange="-25 25"/>
    <position name="aR5" joint="Rj5" kp="120" ctrlrange="-170 170" forcerange="-10 10"/>
    <position name="aR6" joint="Rj6" kp="150" ctrlrange="-120 120" forcerange="-10 10"/>
    <position name="aR7" joint="Rj7" kp="80"  ctrlrange="-170 170" forcerange="-6 6"/>
    <position name="aRfA" joint="RfAs" kp="60" ctrlrange="0 0.026" forcerange="-3 3"/>
    <position name="aRfB" joint="RfBs" kp="60" ctrlrange="0 0.026" forcerange="-3 3"/>
  </actuator>
</mujoco>'''
    return xml, n_actual

if __name__ == "__main__":
    xml, n = build_xml()
    with open("scene.xml", "w") as f:
        f.write(xml)
    print("scene.xml written,", n, "particles")
