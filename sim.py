"""GRID-style dual-arm lab pour, rebuilt in MuJoCo.

Recreation of General Robotics' Auto-Engineering demo (X, Sep 9 2026):
left arm picks a test tube from the rack, hands it to the right arm,
right arm pours pink liquid into a beaker. Liquid = real particles,
beaker fill rises by volume conservation. Renders a 26s 720p video.

Run:  xvfb-run -a python3 sim.py
"""
import numpy as np, mujoco, math, subprocess, sys, os

DEG = math.pi / 180
SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene.xml")
OUT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.mp4")

# scene constants (mirror gen_scene.py)
BENCH_Z, TUBE_H, BEAK_R, BEAK_H = 0.75, 0.13, 0.028, 0.08
TUBE_C0 = np.array([-0.38, -0.12, BENCH_Z + 0.002 + TUBE_H/2])
BEAK    = np.array([0.30, -0.14, BENCH_Z])
P_RAD   = 0.0028
POUR_TUBE_C = np.array([0.202, -0.14, 0.8896])
TILT_AXIS   = np.array([math.sin(103*DEG), 0, math.cos(103*DEG)])

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)

# ---------- helpers ----------
def jid(name):  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
def bid(name):  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
def sid(name):  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
def aid(name):  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
def eid(name):  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, name)

ARM_J = {p: [jid(f"{p}j{k}") for k in range(1, 8)] for p in "LR"}
ARM_Q = {p: [m.jnt_qposadr[j] for j in ARM_J[p]] for p in "LR"}
ARM_A = {p: [aid(f"a{p}{k}") for k in range(1, 8)] for p in "LR"}
FING_A = {p: [aid(f"a{p}fA"), aid(f"a{p}fB")] for p in "LR"}
EE = {p: sid(f"{p}ee") for p in "LR"}
JNT_RANGE = m.jnt_range.copy()

def quat_mul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2])
def quat_inv(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def q2m(q):
    M = np.zeros(9); mujoco.mju_quat2Mat(M, q); return M.reshape(3,3)
def m2q(M):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, M.flatten()); return q

def ik_once(p, pos, R, q_seed, iters, lam, tol=2e-4):
    q = q_seed.copy()
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    qads = ARM_Q[p]; jids = ARM_J[p]
    home = q_seed.copy()
    for _ in range(iters):
        for a, qq in zip(qads, q): d.qpos[a] = qq
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, jacp, jacr, EE[p])
        spos = d.site_xpos[EE[p]].copy()
        smat = d.site_xmat[EE[p]].reshape(3,3)
        epos = pos - spos
        erot = 0.5*(np.cross(smat[:,0], R[:,0]) + np.cross(smat[:,1], R[:,1])
                    + np.cross(smat[:,2], R[:,2]))
        err = np.concatenate([epos, 0.7*erot])
        if np.linalg.norm(epos) < tol and np.linalg.norm(erot) < 4*tol:
            break
        J = np.zeros((6, 7))
        for k, j in enumerate(jids):
            dof = m.jnt_dofadr[j]
            J[:3, k] = jacp[:, dof]; J[3:, k] = jacr[:, dof]
        JJt = J @ J.T + (lam**2)*np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        dq += 0.02 * (np.eye(7) - J.T @ np.linalg.solve(JJt, J)) @ (home - q)
        dq = np.clip(dq, -0.35, 0.35)
        q = q + dq
        for k, j in enumerate(jids):
            q[k] = np.clip(q[k], JNT_RANGE[j][0], JNT_RANGE[j][1])
    for a, qq in zip(qads, q): d.qpos[a] = qq
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    return q, np.linalg.norm(d.site_xpos[EE[p]] - pos)

def pose_err(p, pos, R, q):
    for a, qq in zip(ARM_Q[p], q): d.qpos[a] = qq
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    smat = d.site_xmat[EE[p]].reshape(3,3)
    ep = np.linalg.norm(d.site_xpos[EE[p]] - pos)
    er = np.linalg.norm(0.5*(np.cross(smat[:,0],R[:,0]) + np.cross(smat[:,1],R[:,1]) + np.cross(smat[:,2],R[:,2])))
    return ep, er

def ik(p, pos, R, q_seed, tol=6e-3):
    """DLS IK with restarts; score = pos + 0.3*rot + continuity."""
    def score(q, prev):
        ep, er = pose_err(p, pos, R, q)
        return ep + 0.3*er + 0.02*np.linalg.norm(q - prev), ep
    q, err = ik_once(p, pos, R, q_seed, 500, 0.06)
    best, (bs, be) = q, score(q, q_seed)
    rng = np.random.default_rng(3)
    for tries in range(8):
        if be < tol: break
        seed = q_seed + rng.normal(0, 0.3, 7) if tries % 2 else q + rng.normal(0, 0.2, 7)
        q2, err2 = ik_once(p, pos, R, seed, 500, 0.04)
        s2, e2 = score(q2, q_seed)
        if s2 < bs: best, bs, be = q2, s2, e2
    return best

def hand_target(tube_c, axis, approach, grasp_off):
    """Hand pose to hold tube: local x = tube axis, local z = approach dir."""
    x = axis / np.linalg.norm(axis)
    z = approach / np.linalg.norm(approach)
    y = np.cross(z, x); y /= np.linalg.norm(y)
    z = np.cross(x, y)
    R = np.column_stack([x, y, z])
    pos = tube_c + axis*grasp_off   # ee SITE target (site sits at fingertip zone)
    return pos, R

Z = np.array([0.,0.,1.]); X = np.array([1.,0.,0.])
APPR = {'L': X, 'R': -X}   # approach directions

def minjerk(t):
    t = np.clip(t, 0, 1); return 10*t**3 - 15*t**4 + 6*t**5

# ---------- keyframe plan ----------
HOME = {'L': np.array([25, 42, 0, 65, 0, -25, 0])*DEG,
        'R': np.array([25, 42, 0, 65, 0, -25, 0])*DEG}

# tube waypoints -> hand targets
def L_pose(tube_c, off=0.03, axis=Z):  return hand_target(tube_c, axis, APPR['L'], off)
def R_pose(tube_c, off=-0.03, axis=Z): return hand_target(tube_c, axis, APPR['R'], off)

HANDOFF_C = np.array([-0.12, 0.02, 0.95])

# keyframes: (t, L_spec, R_spec)  spec = (pos, R) or None=hold
keys = []
def add(t, L=None, R=None): keys.append((t, L, R))

pre_c   = TUBE_C0 + np.array([0, 0, 0.10])
lift_c  = TUBE_C0 + np.array([0, 0, 0.19])
pourTilt_R = np.column_stack([TILT_AXIS, np.cross(np.cross(TILT_AXIS, np.array([0.,1.,0.])), TILT_AXIS)/1.0, np.cross(TILT_AXIS, np.array([0.,1.,0.]))])
# build tilt hand frame explicitly
zt = TILT_AXIS / np.linalg.norm(TILT_AXIS)
zt2 = np.cross(zt, np.array([0.,1.,0.])); zt2 /= np.linalg.norm(zt2)
yt2 = np.cross(zt2, zt)
R_tube_tilt = np.column_stack([zt, yt2, zt2])  # local x = tilted axis

add(0.0,  None, None)                                   # home
add(3.0,  L_pose(pre_c), None)                          # pre-grasp
add(4.2,  L_pose(TUBE_C0), None)                        # grasp
add(6.2,  L_pose(lift_c), None)                         # lift
add(8.8,  L_pose(HANDOFF_C), None)                      # handoff point
add(10.8, None, R_pose(HANDOFF_C))                      # right grasps
add(12.8, (np.array([-0.30, 0.18, 1.05]),
           np.column_stack([Z, np.array([0.,1.,0.]), X])), None)   # left retracts
add(15.0, None, R_pose(POUR_TUBE_C))                    # carry over beaker
add(16.6, None, R_pose(POUR_TUBE_C, axis=zt))           # tilt
add(19.9, None, R_pose(POUR_TUBE_C, axis=zt))           # hold pour
add(21.3, None, R_pose(POUR_TUBE_C))                    # tilt back
add(23.2, None, R_pose(HANDOFF_C))                      # back to handoff
add(24.6, L_pose(HANDOFF_C), None)                      # left grasps again
add(26.8, L_pose(TUBE_C0), None)                        # tube back to rack
add(28.2, (np.array([-0.30, 0.18, 1.05]),
           np.column_stack([Z, np.array([0.,1.,0.]), X])), None)   # left retracts
T_END = 29.0

# solve IK at each keyframe
print("solving IK ...")
qK = []
qL, qR = HOME['L'].copy(), HOME['R'].copy()
for t, Ls, Rs in keys:
    if Ls is not None: qL = ik('L', Ls[0], Ls[1], qL)
    if Rs is not None: qR = ik('R', Rs[0], Rs[1], qR)
    qK.append((t, qL.copy(), qR.copy()))
    # report residual
    for p, s, q in (('L', Ls, qL), ('R', Rs, qR)):
        if s is None: continue
        for a, qq in zip(ARM_Q[p], q): d.qpos[a] = qq
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        ep = np.linalg.norm(d.site_xpos[EE[p]] - s[0])
        print(f"  t={t:5.1f} {p} err={ep*1000:6.1f}mm")

# ---------- events ----------
EQ = {n: eid(n) for n in ("world_tube", "L_tube", "R_tube")}
PW = []  # particles ride free in the tube

def set_weld(name, body1, active, snap_pose=None):
    i = EQ[name]
    if active:
        if snap_pose is not None:
            m.eq_data[i][3:6] = snap_pose[0]; m.eq_data[i][6:10] = snap_pose[1]
        else:
            b1, b2 = (bid(body1) if body1 else 0), bid("tube")
            x1 = d.xpos[b1].copy() if b1 else np.zeros(3)
            q1 = d.xquat[b1].copy() if b1 else np.array([1.,0,0,0])
            R1 = q2m(q1)
            relpos = R1.T @ (d.xpos[b2] - x1)
            relq = quat_mul(quat_inv(q1), d.xquat[b2])
            m.eq_data[i][3:6] = relpos; m.eq_data[i][6:10] = relq
    d.eq_active[i] = 1 if active else 0

# finger events: (t, L_tgt, R_tgt)  0=open 0.014=closed
fing_ev = [(4.2, 0.014, None), (10.8, None, 0.014),
           (11.5, 0.0, None), (24.6, 0.014, None), (25.1, None, 0.0),
           (27.0, 0.0, None)]
# planned grasp relposes: relpos = (-grasp_off, 0, 0.115), relquat = R_hand^T
def planned_snap(pose_fn, tube_c, off):
    _, Rh = pose_fn(tube_c, off=off)
    return (np.array([-off, 0.0, 0.115]), m2q(Rh.T))
SNAP_L  = planned_snap(L_pose, TUBE_C0, 0.03)
SNAP_R  = planned_snap(R_pose, HANDOFF_C, -0.03)
SNAP_L2 = planned_snap(L_pose, HANDOFF_C, 0.03)
SNAP_W  = (TUBE_C0.copy(), np.array([1., 0, 0, 0]))
# weld events: (t, name, active, snap_pose_or_None)
weld_ev = [(4.8, "L_tube", True, SNAP_L),  (4.9, "world_tube", False, None),
           (11.3, "R_tube", True, SNAP_R), (11.4, "L_tube", False, None),
           (24.9, "L_tube", True, SNAP_L2), (25.0, "R_tube", False, None),
           (26.8, "world_tube", True, SNAP_W), (26.9, "L_tube", False, None)]
UNWELD_T0, UNWELD_SPAN = 16.7, 2.2   # staggered particle release
POUR_DONE_T = 19.9

fing_ev.sort(); weld_ev.sort()
fi = wi = 0
unweld_started = False
absorbed = 0
V_p = 4/3*math.pi*P_RAD**3
A_b = math.pi*(BEAK_R-0.002)**2
liq_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "beak_liquid")
NPART = 0
while bid(f"p{NPART}") >= 0:
    NPART += 1
PART_BODIES = [bid(f"p{i}") for i in range(NPART)]
PART_Q = [m.jnt_qposadr[jid(f"pf{i}")] for i in range(NPART)]
PART_DOF = [m.jnt_dofadr[jid(f"pf{i}")] for i in range(NPART)]

# ---------- reset ----------
mujoco.mj_resetData(m, d)
for p in "LR":
    for a, qq in zip(ARM_Q[p], HOME[p]): d.qpos[a] = qq
    d.ctrl[ARM_A[p]] = HOME[p]
    d.ctrl[FING_A[p]] = [0.0, 0.0]
d.eq_active[EQ["world_tube"]] = 1
mujoco.mj_forward(m, d)
_tube_b0 = bid("tube")
PART_LOCAL = np.array([(d.xpos[b] - d.xpos[_tube_b0]).copy() for b in PART_BODIES])  # tube upright at t=0

# ---------- camera / video ----------
r = mujoco.Renderer(m, 720, 1280)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance = 1.62; cam.elevation = 13
ff = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", "1280x720", "-r", "30", "-i", "-", "-an", "-c:v", "libx264",
    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

FPS, DT = 30, m.opt.timestep
steps_per_frame = int(round((1.0/FPS)/DT))
nframes = int(T_END*FPS)
posters = {13: "frame_carry.jpg", 9: "frame_handoff.jpg", 18: "frame_pour.jpg"}

def targets_at(t):
    """Interpolated joint targets for both arms at time t."""
    out = {}
    for arm_i, p in enumerate("LR"):
        for k in range(len(qK)-1):
            t0, t1 = qK[k][0], qK[k+1][0]
            if t0 <= t <= t1:
                s = minjerk((t-t0)/(t1-t0)) if t1 > t0 else 1.0
                out[p] = (1-s)*qK[k][arm_i+1] + s*qK[k+1][arm_i+1]
                break
        else:
            out[p] = qK[-1][arm_i+1]
    return out

print("simulating ...")
frame = 0
for f in range(nframes):
    t = f / FPS
    # events
    while fi < len(fing_ev) and fing_ev[fi][0] <= t:
        _, Lv, Rv = fing_ev[fi]
        if Lv is not None: d.ctrl[FING_A['L']] = [Lv, Lv]
        if Rv is not None: d.ctrl[FING_A['R']] = [Rv, Rv]
        fi += 1
    while wi < len(weld_ev) and weld_ev[wi][0] <= t:
        _, nm, act, snap = weld_ev[wi]
        set_weld(nm, "Lhand" if nm == "L_tube" else ("Rhand" if nm == "R_tube" else ""), act, snap)
        wi += 1
    # staggered particle release during pour
    # joint targets
    tg = targets_at(t)
    for p in "LR": d.ctrl[ARM_A[p]] = tg[p]
    # step
    for _ in range(steps_per_frame):
        mujoco.mj_step(m, d)
        tb_ = bid("tube")
        Rt_ = q2m(d.xquat[tb_])
        for i, ((bi, q0, dof), loc) in enumerate(zip(zip(PART_BODIES, PART_Q, PART_DOF), PART_LOCAL)):
            if t < UNWELD_T0 + UNWELD_SPAN*i/len(PART_BODIES):  # pinned in tube until its release slot
                d.qpos[q0:q0+3] = d.xpos[tb_] + Rt_ @ loc
                d.qpos[q0+3:q0+7] = [1, 0, 0, 0]
                d.qvel[dof:dof+6] = 0
        # absorption
        if t > UNWELD_T0:
            for bi, q0, dof in zip(PART_BODIES, PART_Q, PART_DOF):
                if d.qpos[q0+2] < 0: continue
                px, py, pz = d.xpos[bi]
                dx, dy = px-BEAK[0], py-BEAK[1]
                if dx*dx + dy*dy < (BEAK_R-0.004)**2 and pz < BENCH_Z + BEAK_H - 0.006:
                    d.qpos[q0:q0+3] = [2.5 + 0.008*(bi % 50), 2.5, 0.004]
                    d.qpos[q0+3:q0+7] = [1, 0, 0, 0]
                    d.qvel[dof:dof+6] = 0
                    absorbed += 1
                    h = absorbed*V_p/A_b
                    m.geom_size[liq_gid][1] = max(h/2, 0.0001)
                    m.geom_pos[liq_gid][2] = 0.003 + h/2
    # render
    cam.azimuth = -102 + 26*(f/nframes)
    p_in = min(max((t-15.0)/2.0, 0.0), 1.0); p_out = min(max((t-20.5)/2.5, 0.0), 1.0)
    p = (p_in*p_in*(3-2*p_in)) * (1 - p_out*p_out*(3-2*p_out))
    cam.lookat[:] = (1-p)*np.array([0.02, -0.06, 0.93]) + p*np.array([0.22, -0.12, 0.90])
    cam.distance = (1-p)*1.62 + p*1.12
    cam.elevation = (1-p)*13 + p*24
    r.update_scene(d, camera=cam)
    ff.stdin.write(r.render().tobytes())
    if f in posters.values() and False: pass
    for pt, name in posters.items():
        if f == pt*FPS:
            import imageio.v2 as imageio
            imageio.imwrite(os.path.join(os.path.dirname(OUT), name), r.render())
    frame += 1
    if f % 90 == 0: print(f"  t={t:5.1f} absorbed={absorbed}")

ff.stdin.close(); ff.wait()
print(f"done. absorbed {absorbed}/{len(PW)} particles -> {OUT}")
