import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, interp1d
import math
import numpy as np
import random, openpyxl

# ---------- Load track from Excel ----------
file_path = Path("Skitpad.xlsx")
if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_excel(file_path, engine="openpyxl")
numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
if len(numeric_cols) < 2:
    raise ValueError("Excel must contain at least two numeric columns (x, y).")
x_col, y_col = numeric_cols[0], numeric_cols[1]
raw_x = df[x_col].to_numpy()
raw_y = df[y_col].to_numpy()

# ---------- Densify path using B-spline ----------
dx = np.diff(raw_x); dy = np.diff(raw_y)
seglen = np.hypot(dx, dy)
s = np.concatenate([[0], np.cumsum(seglen)])
t_raw = s / s[-1]

try:
    tck, _ = splprep([raw_x, raw_y], u=t_raw, s=0.0, k=min(3, len(raw_x)-1))
    u_dense = np.linspace(0, 1, max(1000, len(raw_x)*5))
    x_dense, y_dense = splev(u_dense, tck)
    dx1, dy1 = splev(u_dense, tck, der=1)
    dx2, dy2 = splev(u_dense, tck, der=2)
    denom = (np.hypot(dx1, dy1)**3) + 1e-9
    kappa = (dx1*dy2 - dy1*dx2)/denom
except Exception:
    # fallback to linear interpolation
    u_dense = np.linspace(0, 1, max(2000, len(raw_x)*5))
    x_dense = np.interp(u_dense, t_raw, raw_x)
    y_dense = np.interp(u_dense, t_raw, raw_y)
    dx1 = np.gradient(x_dense); dy1 = np.gradient(y_dense)
    dx2 = np.gradient(dx1); dy2 = np.gradient(dy1)
    denom = (np.hypot(dx1, dy1)**3) + 1e-9
    kappa = (dx1*dy2 - dy1*dx2)/denom

abs_kappa = np.abs(kappa)
s_dense = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x_dense), np.diff(y_dense)))])

# ---------- Vehicle & Controller Parameters ----------
L = 2.0             
dt = 0.05           # smaller timestep for faster car
v_cruise = 30.0     # m/s (~108 km/h)
steer_limit = 0.5
base_lookahead = 5.0
min_lookahead = 2.0
max_lookahead = 15.0

v_min, v_max = 5.0, v_cruise
kappa_speed_scale = 5.0
accel_max = 5.0
n_candidates = 20
preview_T = 0.5
sim_preview_steps = int(1.0/dt)

# ---------- Helper functions ----------
def wrap_angle(a): return (a+np.pi) % (2*np.pi) - np.pi

def kinematic_bicycle_step(state, delta, v):
    x, y, yaw = state
    x += v * math.cos(yaw) * dt
    y += v * math.sin(yaw) * dt
    yaw += v/L * math.tan(delta) * dt
    return np.array([x, y, wrap_angle(yaw)])

def closest_index_on_path(px, py, path_x, path_y, start_idx=0):
    dx = path_x[start_idx:] - px
    dy = path_y[start_idx:] - py
    dist = np.hypot(dx, dy)
    idx = np.argmin(dist) + start_idx
    return idx, dist[idx - start_idx]

def curvature_adaptive_lookahead(k_local):
    look = base_lookahead / (1.0 + 10*k_local)
    return np.clip(look, min_lookahead, max_lookahead)

def velocity_profile_from_curvature(abs_kappa):
    v_profile = v_max / (1 + kappa_speed_scale*abs_kappa)
    return np.clip(v_profile, v_min, v_max)

def pure_pursuit_control(state, path_x, path_y, start_idx, lookahead_m):
    x, y, yaw = state
    idx = start_idx
    while idx < len(path_x)-1:
        d = math.hypot(path_x[idx+1]-x, path_y[idx+1]-y)
        if d >= lookahead_m:
            idx += 1
            break
        idx += 1
    idx = min(idx, len(path_x)-1)
    gx, gy = path_x[idx], path_y[idx]
    alpha = wrap_angle(math.atan2(gy-y, gx-x) - yaw)
    Ld = max(1e-3, math.hypot(gx-x, gy-y))
    delta = math.atan2(2*L*math.sin(alpha), Ld)
    return np.clip(delta, -steer_limit, steer_limit), idx

def score_candidate(state, path_x, path_y, start_idx, lookahead, v_local):
    sim_state = state.copy()
    idx = start_idx
    cte_total = 0.0
    for _ in range(sim_preview_steps):
        delta, idx = pure_pursuit_control(sim_state, path_x, path_y, idx, lookahead)
        sim_state = kinematic_bicycle_step(sim_state, delta, v_local)
        _, d_closest = closest_index_on_path(sim_state[0], sim_state[1], path_x, path_y, start_idx)
        cte_total += d_closest
    return cte_total

# ---------- Velocity profile ----------
v_profile = velocity_profile_from_curvature(abs_kappa)
v_profile_smooth = np.convolve(v_profile, np.ones(max(3,int(0.5/dt)))/max(3,int(0.5/dt)), mode='same')
v_along_s = interp1d(s_dense, v_profile_smooth, bounds_error=False, fill_value=(v_profile_smooth[0], v_profile_smooth[-1]))

# ---------- Simulation Loop ----------
state = np.array([x_dense[0], y_dense[0], math.atan2(y_dense[1]-y_dense[0], x_dense[1]-x_dense[0])])
trajectory = [state.copy()]
steer_hist, vel_hist, lookahead_hist, cte_hist = [], [], [], []
start_idx = 0

while True:
    start_idx, _ = closest_index_on_path(state[0], state[1], x_dense, y_dense, start_idx)

    k_local = np.max(abs_kappa[start_idx: min(start_idx+10, len(abs_kappa))])
    baseline_look = curvature_adaptive_lookahead(k_local)

    s_here = s_dense[start_idx]
    v_here = float(v_along_s(s_here))
    v_prev = vel_hist[-1] if vel_hist else v_here
    v_here = float(np.clip(v_here, v_prev - accel_max*dt, v_prev + accel_max*dt))

    # generate candidate lookahead distances
    candidates = []
    for _ in range(n_candidates):
        v_sample = float(np.clip(v_here, v_min, v_max))
        look = np.clip(v_sample*preview_T, min_lookahead, max_lookahead)
        look = 0.6*look + 0.4*baseline_look
        candidates.append((look, v_sample))

    # pick optimal lookahead
    best_score = float('inf'); best_look = baseline_look
    for look, v_cand in candidates:
        score = score_candidate(state, x_dense, y_dense, start_idx, look, v_cand)
        if score < best_score:
            best_score = score
            best_look = look

    delta, idx_look = pure_pursuit_control(state, x_dense, y_dense, start_idx, best_look)
    state = kinematic_bicycle_step(state, delta, v_here)

    trajectory.append(state.copy())
    steer_hist.append(delta)
    vel_hist.append(v_here)
    lookahead_hist.append(best_look)
    _, d_closest = closest_index_on_path(state[0], state[1], x_dense, y_dense, start_idx)
    cte_hist.append(d_closest)

    if np.hypot(state[0]-x_dense[-1], state[1]-y_dense[-1]) < 1.0:
        break

trajectory = np.array(trajectory)
steer_hist = np.array(steer_hist)
vel_hist = np.array(vel_hist)
lookahead_hist = np.array(lookahead_hist)
cte_hist = np.array(cte_hist)

# ---------- Compute cumulative distance along trajectory ----------
traj_dx = np.diff(trajectory[:,0][:len(steer_hist)], prepend=trajectory[0,0])
traj_dy = np.diff(trajectory[:,1][:len(steer_hist)], prepend=trajectory[0,1])
traj_dist = np.cumsum(np.hypot(traj_dx, traj_dy))

# ---------- Plot results vs distance ----------
plt.figure(figsize=(10,6))
plt.plot(raw_x, raw_y, 'k.-', alpha=0.4, label='Raw waypoints')
plt.plot(x_dense, y_dense, 'g-', label='Dense B-spline path')
plt.plot(trajectory[:,0], trajectory[:,1], 'b-', linewidth=2, label='Vehicle trajectory')
plt.scatter(x_dense[0], y_dense[0], c='r', label='Start')
plt.scatter(x_dense[-1], y_dense[-1], c='m', label='Goal')
plt.axis('equal'); plt.legend(); plt.grid(True); plt.title('Trajectory')

plt.figure(figsize=(10,3))
plt.plot(traj_dist, steer_hist, label='Steering (rad)')
plt.plot(traj_dist, lookahead_hist, label='Lookahead (m)')
plt.xlabel('Distance traveled (m)'); plt.ylabel('Value')
plt.grid(True); plt.legend(); plt.title('Steering & Lookahead vs Distance')

plt.figure(figsize=(10,3))
plt.plot(traj_dist, vel_hist, label='Velocity (m/s)')
plt.xlabel('Distance traveled (m)'); plt.ylabel('Velocity (m/s)')
plt.grid(True); plt.legend(); plt.title('Velocity vs Distance')

plt.figure(figsize=(10,3))
plt.plot(traj_dist, cte_hist, label='Cross-track error (m)')
plt.xlabel('Distance traveled (m)'); plt.ylabel('Error (m)')
plt.grid(True); plt.legend(); plt.title('Cross-track Error vs Distance')

plt.show()