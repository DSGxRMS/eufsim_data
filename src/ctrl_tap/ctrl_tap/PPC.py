import time
import math
import numpy as np
import pandas as pd
import fsds
from scipy.interpolate import splprep, splev

# -------------------- Setup --------------------
client = fsds.FSDSClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)

PATHPOINTS_CSV = "D:\\RMS\\FSDS\\impscripts\\pathpoint.csv"
ROUTE_IS_LOOP = False
scaling_factor = 1

SEARCH_BACK = 10
SEARCH_FWD = 250
MAX_STEP = 60

WHEELBASE_M = 1.5
MAX_STEER_RAD = 0.2
LD_BASE = 3.5
LD_GAIN = 0.6
LD_MIN = 2.0
LD_MAX = 15.0

V_MAX = 8.0
AY_MAX = 4.0
AX_MAX = 5.0
AX_MIN = -4.0

PROFILE_WINDOW_M = 100.0
BRAKE_EXTEND_M = 60.0
NUM_ARC_POINTS = 800
PROFILE_HZ = 10.0
BRAKE_GAIN = 0.7

STOP_SPEED_THRESHOLD = 0.1   # m/s, vehicle considered stopped

# Jerk-limited velocity profile params (from Controls_final.m)
V_MIN = 5.0          # m/s
A_MAX = 15.0         # m/s^2
D_MAX = 20.0         # m/s^2 (max decel)
J_MAX = 70.0         # m/s^3
CURVATURE_MAX = 0.9  # 1/m

# -------------------- Utility Functions --------------------
def preprocess_path(xs, ys, loop=True):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x_next = np.roll(xs, -1) if loop else np.concatenate((xs[1:], xs[-1:]))
    y_next = np.roll(ys, -1) if loop else np.concatenate((ys[1:], ys[-1:]))
    seglen = np.hypot(x_next - xs, y_next - ys)
    if not loop:
        seglen[-1] = 0.0
    s = np.concatenate(([0.0], np.cumsum(seglen[:-1])))
    return xs, ys, s, float(seglen.sum())

def get_xy_speed(state):
    pos = state.kinematics_estimated.position
    return (pos.x_val, pos.y_val), float(getattr(state, "speed", 0.0))

def local_closest_index(xy, xs, ys, cur_idx, loop=True):
    x0, y0 = xy
    N = len(xs)
    if N == 0:
        return 0
    if loop:
        start = (cur_idx - SEARCH_BACK) % N
        count = min(N, SEARCH_BACK + SEARCH_FWD + 1)
        idxs = (np.arange(start, start + count) % N)
        dx, dy = xs[idxs] - x0, ys[idxs] - y0
        j = int(np.argmin(dx * dx + dy * dy))
        return int(idxs[j])
    else:
        i0 = max(0, cur_idx - SEARCH_BACK)
        i1 = min(N, cur_idx + SEARCH_FWD + 1)
        dx, dy = xs[i0:i1] - x0, ys[i0:i1] - y0
        j = int(np.argmin(dx * dx + dy * dy))
        return i0 + j

def calc_lookahead(speed_mps):
    return max(LD_MIN, min(LD_MAX, LD_BASE + LD_GAIN * speed_mps))

def get_yaw(state):
    q = state.kinematics_estimated.orientation
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def forward_index_by_distance(near_idx, Ld, s, total_len, loop=True):
    if loop:
        target = (s[near_idx] + Ld) % total_len
        return int(np.searchsorted(s, target, side="left") % len(s))
    else:
        target = min(s[near_idx] + Ld, s[-1])
        return int(np.searchsorted(s, target, side="left"))

def pure_pursuit_steer(pos_xy, yaw, speed, xs, ys, near_idx, s, total_len, loop=True):
    Ld = calc_lookahead(speed)
    tgt_idx = forward_index_by_distance(near_idx, Ld, s, total_len, loop)
    tx, ty = xs[tgt_idx], ys[tgt_idx]
    dx, dy = tx - pos_xy[0], ty - pos_xy[1]
    cy, sy = math.cos(yaw), math.sin(yaw)
    x_rel, y_rel = cy * dx + sy * dy, -sy * dx + cy * dy
    kappa = 2.0 * y_rel / max(0.5, Ld) ** 2
    delta = math.atan(WHEELBASE_M * kappa)
    return max(-1, min(1, delta / MAX_STEER_RAD)), tgt_idx

def resample_track(x_raw, y_raw, num_arc_points=NUM_ARC_POINTS):
    tck, _ = splprep([x_raw, y_raw], s=0, k=min(3, max(1, len(x_raw) - 1)))
    tt_dense = np.linspace(0, 1, 2000)
    xx, yy = splev(tt_dense, tck)
    s_dense = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))))
    s_dense /= s_dense[-1] if s_dense[-1] > 0 else 1.0
    s_uniform = np.linspace(0, 1, num_arc_points)
    return np.interp(s_uniform, s_dense, xx), np.interp(s_uniform, s_dense, yy)

def compute_curvature(x, y):
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = np.power(dx*dx + dy*dy, 1.5)
    curv = np.abs(dx * ddy - dy * ddx) / (denom + 1e-12)
    curv[~np.isfinite(curv)] = 0.0
    return curv

def curvature_speed_limit(curvature):
    return np.minimum(np.sqrt(AY_MAX / (curvature + 1e-9)), V_MAX)

def profile_window(v_limit_win, ds_win, v0):
    Nw = len(v_limit_win)
    vp = np.zeros(Nw)
    vp[0] = min(v_limit_win[0], v0)
    for i in range(1, Nw):
        vp[i] = min(math.sqrt(vp[i-1]**2 + 2 * AX_MAX * ds_win[i-1]), v_limit_win[i])
    for i in range(Nw - 2, -1, -1):
        vp[i] = min(vp[i], math.sqrt(vp[i+1]**2 + 2 * abs(AX_MIN) * ds_win[i]), v_limit_win[i])
    return vp

def compute_signed_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.power(dx * dx + dy * dy, 1.5) + 1e-12
    kappa = (dx * ddy - dy * ddx) / denom
    kappa = np.clip(kappa, -CURVATURE_MAX, CURVATURE_MAX)
    kappa[~np.isfinite(kappa)] = 0.0
    return kappa

def ackermann_curv_speed_limit(kappa):
    delta = np.arctan(kappa * WHEELBASE_M)
    denom = np.abs(np.tan(delta)) + 1e-6
    v = np.sqrt(np.maximum(0.0, (D_MAX * WHEELBASE_M) / denom))
    return np.minimum(v, V_MAX)

def segment_distances(xs, ys, loop=True):
    x_next = np.roll(xs, -1)
    y_next = np.roll(ys, -1)
    ds = np.hypot(x_next - xs, y_next - ys)
    if not loop:
        ds[-1] = 0.0
    return ds

def jerk_limited_velocity_profile(v_limit, ds, v0, vf, v_min, v_max, a_max, d_max, j_max):
    v_limit = np.asarray(v_limit, dtype=float)
    ds = np.asarray(ds, dtype=float)
    N = len(v_limit)
    if len(ds) != N:
        raise ValueError("ds length must equal v_limit length (ds[0] is 0 for the first point)")
    v_forward = np.zeros(N, dtype=float)
    v_forward[0] = min(max(v0, 0.0), v_limit[0], v_max)
    a_prev = 0.0
    for i in range(1, N):
        ds_i = max(ds[i], 1e-9)
        v_avg = max(v_min, v_forward[i - 1])
        dt = ds_i / v_avg
        a_curr = min(a_prev + j_max * dt, a_max)
        v_possible = math.sqrt(max(0.0, v_forward[i - 1] ** 2 + 2.0 * a_curr * ds_i))
        v_forward[i] = min(v_possible, v_limit[i], v_max)
        a_prev = (v_forward[i] ** 2 - v_forward[i - 1] ** 2) / (2.0 * ds_i)
    v_profile = v_forward.copy()
    v_profile[-1] = min(v_profile[-1], max(0.0, vf))
    a_prev = 0.0
    for i in range(N - 2, -1, -1):
        ds_i = max(ds[i + 1], 1e-9)
        v_avg = max(v_min, v_profile[i + 1])
        dt = ds_i / v_avg
        a_curr = min(a_prev + j_max * dt, d_max)
        v_possible = math.sqrt(max(0.0, v_profile[i + 1] ** 2 + 2.0 * a_curr * ds_i))
        v_profile[i] = min(v_profile[i], v_possible, v_max)
        a_prev = (v_profile[i + 1] ** 2 - v_profile[i] ** 2) / (2.0 * ds_i)
    v_profile = np.minimum(v_profile, v_max)
    if N > 1:
        v_profile[:-1] = np.maximum(v_profile[:-1], v_min)
    if vf <= 0.0:
        v_profile[-1] = 0.0
    return v_profile

class PID:
    # Throttle PID (kept as-is; output clamped to [0,1])
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._i = 0.0
        self._prev_err = None
    def reset(self):
        self._i, self._prev_err = 0.0, None
    def update(self, err, dt):
        dt = max(dt, 1e-3)
        self._i += err * dt
        d = 0 if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err
        return max(0, min(1, self.kp * err + self.ki * self._i + self.kd * d))

class PIDRange:
    # Generic PID with symmetric output limits (for steering correction)
    def __init__(self, kp, ki, kd, out_min=-1.0, out_max=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self._i = 0.0
        self._prev_err = None
    def reset(self):
        self._i, self._prev_err = 0.0, None
    def update(self, err, dt):
        dt = max(dt, 1e-3)
        self._i += err * dt
        d = 0.0 if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err
        u = self.kp * err + self.ki * self._i + self.kd * d
        return max(self.out_min, min(self.out_max, u))

def path_heading(xs, ys, idx, loop=True):
    n = len(xs)
    i2 = (idx + 1) % n if loop else min(idx + 1, n - 1)
    dx = xs[i2] - xs[idx]
    dy = ys[i2] - ys[idx]
    return math.atan2(dy, dx)

def cross_track_error(cx, cy, xs, ys, idx, loop=True):
    # Signed lateral error relative to path tangent at idx
    theta_ref = path_heading(xs, ys, idx, loop)
    dx = cx - xs[idx]
    dy = cy - ys[idx]
    # Rotate world error into path frame: y' is lateral error
    e_lat = -math.sin(theta_ref) * dx + math.cos(theta_ref) * dy
    return e_lat, theta_ref

# -------------------- Load path --------------------
df = pd.read_csv(PATHPOINTS_CSV)
rx, ry = resample_track(df["x"].to_numpy() * scaling_factor,
                        df["y"].to_numpy() * scaling_factor)
route_x, route_y = ry + 15.0, -rx
route_x, route_y, route_s, route_len = preprocess_path(route_x, route_y, loop=ROUTE_IS_LOOP)
seg_ds = segment_distances(route_x, route_y, ROUTE_IS_LOOP)

# Curvature-based speed limit using Ackermann mapping (Controls_final.m)
kappa_signed = compute_signed_curvature(route_x, route_y)
v_limit_global = ackermann_curv_speed_limit(kappa_signed)
route_v = v_limit_global.copy()

# -------------------- Controllers --------------------
th_pid = PID(3.2, 0.5, 0.134)  # throttle
steer_pid = PIDRange(kp=0.15, ki=0.05, kd=0.20, out_min=-0.60, out_max=0.60)  # steering correction

# -------------------- Control Loop --------------------
state = client.getCarState()
(cur_x, cur_y), speed = get_xy_speed(state)
cur_idx = int(np.argmin((route_x - cur_x)**2 + (route_y - cur_y)**2))

last_t = time.perf_counter()
last_profile_t = last_t

log = []
input("Press Enter to start...")
while True:
    state = client.getCarState()
    (cx, cy), speed = get_xy_speed(state)
    yaw = get_yaw(state)

    now = time.perf_counter()
    dt = now - last_t
    last_t = now

    cur_idx = local_closest_index((cx, cy), route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)

    # Velocity profile refresh
    if now - last_profile_t >= (1.0 / PROFILE_HZ):
        last_profile_t = now
        end_idx = forward_index_by_distance(cur_idx, PROFILE_WINDOW_M, route_s, route_len, ROUTE_IS_LOOP)

        if ROUTE_IS_LOOP:
            if end_idx >= cur_idx:
                idxs = np.arange(cur_idx, end_idx + 1)
            else:
                idxs = np.concatenate((np.arange(cur_idx, len(route_x)), np.arange(0, end_idx + 1)))
        else:
            idxs = np.arange(cur_idx, end_idx + 1)

        if ROUTE_IS_LOOP:
            ds_win_list = [0.0]
            for ii in range(len(idxs) - 1):
                ds_win_list.append(seg_ds[idxs[ii]])
            ds_win = np.asarray(ds_win_list, dtype=float)
        else:
            ds_win = np.concatenate(([0.0], seg_ds[idxs[:-1]]))

        v_lim_win = v_limit_global[idxs]
        v0 = speed
        vf = 0.0 if (not ROUTE_IS_LOOP and idxs[-1] == len(route_x) - 1) else float(v_lim_win[-1])

        v_prof_win = jerk_limited_velocity_profile(
            v_lim_win, ds_win, v0, vf, V_MIN, V_MAX, A_MAX, D_MAX, J_MAX
        )

        if ROUTE_IS_LOOP and end_idx < cur_idx:
            n1 = len(route_x) - cur_idx
            route_v[cur_idx:] = v_prof_win[:n1]
            route_v[:end_idx + 1] = v_prof_win[n1:]
        else:
            route_v[cur_idx:end_idx + 1] = v_prof_win

    # --- Pure Pursuit feedforward steering ---
    steering_ppc, tgt_idx = pure_pursuit_steer((cx, cy), yaw, speed,
                                               route_x, route_y, cur_idx,
                                               route_s, route_len, loop=ROUTE_IS_LOOP)

    # --- Lateral error feedback (PID on cross-track error) ---
    e_lat, theta_ref = cross_track_error(cx, cy, route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
    steer_correction = steer_pid.update(e_lat, dt)

    # --- Final steering command ---
    steering_cmd = steering_ppc + steer_correction
    steering_cmd = max(-1.0, min(1.0, steering_cmd))

    # Longitudinal control
    v_err = route_v[cur_idx] - speed
    if v_err >= 0:
        throttle = th_pid.update(v_err, dt)
        brake = 0
    else:
        th_pid.reset()
        throttle, brake = 0, min(1, -v_err * BRAKE_GAIN)

    # Apply to car (keep your scaling/sign)
    car_controls = fsds.CarControls()
    car_controls.throttle = throttle
    car_controls.brake = brake
    car_controls.steering = -steering_cmd * 1.2
    client.setCarControls(car_controls)

    # --- Logging (add PPC vs correction vs final, and refs) ---
    tgt_x, tgt_y = route_x[tgt_idx], route_y[tgt_idx]

    Ld = calc_lookahead(speed)
    dx_t, dy_t = tgt_x - cx, tgt_y - cy
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    x_rel, y_rel = cyaw * dx_t + syaw * dy_t, -syaw * dx_t + cyaw * dy_t
    kappa_ppc = 2.0 * y_rel / max(0.5, Ld) ** 2

    # Ackermann reference from path curvature at current index (optional, for analysis)
    kappa_ref = kappa_signed[cur_idx]
    steering_ack = math.atan(WHEELBASE_M * kappa_ref) / MAX_STEER_RAD
    steering_ack = max(-1.0, min(1.0, steering_ack))

    log.append([
        now, cx, cy, speed, throttle, brake,
        steering_ppc,          # ppc steer (normalized)
        steer_correction,      # pid correction
        steering_cmd,          # final command (normalized)
        v_err, yaw,
        tgt_x, tgt_y,
        kappa_ppc,
        e_lat,                 # cross-track error
        steering_ack           # ackermann reference (normalized)
    ])

    if len(log) % 10 == 0:
        pd.DataFrame(log, columns=[
            "t","x","y","speed","throttle","brake",
            "steer_ppc","steer_corr","steer_cmd",
            "v_err","yaw",
            "tgt_x","tgt_y",
            "kappa_ppc","e_lat","steer_ack"
        ]).to_csv("impscripts\\telemetry_log.csv", index=False)

    # --- Exit condition: vehicle has stopped at end of route ---
    if (not ROUTE_IS_LOOP) and cur_idx >= len(route_x) - 1 and speed < STOP_SPEED_THRESHOLD:
        print("Reached end of route and stopped. Exiting loop.")
        car_controls.handbrake = True
        client.setCarControls(car_controls)
        break

    time.sleep(0.02)
