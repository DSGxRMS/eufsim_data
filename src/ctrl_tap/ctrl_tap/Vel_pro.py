import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, interp1d

# ==============================
# 1. Load and Resample Track
# ==============================
data = pd.read_excel("skitpad.xlsx")
x_raw = data["X"].to_numpy()
y_raw = data["Y"].to_numpy()

# Parametrize original points by cumulative arc length
t_original = np.insert(np.cumsum(np.sqrt(np.diff(x_raw)**2 + np.diff(y_raw)**2)), 0, 0)
t_original = t_original / t_original[-1]

# Fit B-spline
ppX = splrep(t_original, x_raw, k=3)
ppY = splrep(t_original, y_raw, k=3)

# Dense spline sampling
tt_dense = np.linspace(0, 1, 2000)
xx_dense = splev(tt_dense, ppX)
yy_dense = splev(tt_dense, ppY)
s_dense = np.insert(np.cumsum(np.sqrt(np.diff(xx_dense)**2 + np.diff(yy_dense)**2)), 0, 0)
s_dense = s_dense / s_dense[-1]

# Resample to uniform arc length
numArcPoints = 1000
s_uniform = np.linspace(0, 1, numArcPoints)
interp_x = interp1d(s_dense, xx_dense, kind="linear", fill_value="extrapolate")
interp_y = interp1d(s_dense, yy_dense, kind="linear", fill_value="extrapolate")
x = interp_x(s_uniform)
y = interp_y(s_uniform)

# ==============================
# 2. Arc length & curvature
# ==============================
dx = np.gradient(x)
dy = np.gradient(y)
ddx = np.gradient(dx)
ddy = np.gradient(dy)

curvature = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2) ** 1.5 + 1e-9)
curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

s = np.insert(np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)), 0, 0)
N = len(x)

# ==============================
# 3. Vehicle constraints
# ==============================
v_max = 50.0
ay_max = 10.0
ax_max = 12.0
ax_min = -8.0

# ==============================
# 4. Velocity profile
# ==============================
v_curve = np.sqrt(np.maximum(ay_max, 0) / (curvature + 1e-9))
v_limit = np.minimum(v_curve, v_max)

v_profile = np.zeros(N)
v_profile[0] = min(v_limit[0], 1.0)

# Forward pass
for i in range(1, N):
    ds = max(s[i] - s[i-1], 1e-9)
    v_allowed = np.sqrt(max(v_profile[i-1]**2 + 2 * ax_max * ds, 0))
    v_profile[i] = min(v_allowed, v_limit[i])

# Backward pass
for i in range(N-2, -1, -1):
    ds = max(s[i+1] - s[i], 1e-9)
    v_allowed = np.sqrt(max(v_profile[i+1]**2 + 2 * abs(ax_min) * ds, 0))
    v_profile[i] = min(v_profile[i], v_allowed, v_limit[i])

v_profile = np.nan_to_num(v_profile, nan=0.0, posinf=v_max, neginf=0.0)

# ==============================
# 5. Acceleration and jerk
# ==============================
a_profile = np.gradient(v_profile, s, edge_order=2)
jerk_profile = np.gradient(a_profile, s, edge_order=2)

a_profile = np.nan_to_num(a_profile, nan=0.0, posinf=0.0, neginf=0.0)
jerk_profile = np.nan_to_num(jerk_profile, nan=0.0, posinf=0.0, neginf=0.0)

# ==============================
# 6. Visualization
# ==============================
fig, axs = plt.subplots(2, 3, figsize=(14, 8), facecolor="black")
(ax_track, ax_vel, ax_acc), (ax_jerk, ax_curve, ax_empty) = axs

# Track
ax_track.plot(x, y, "k--", linewidth=0.8)
car_marker, = ax_track.plot([], [], "o", markersize=8, color="r")
ax_track.set_title("Track with Moving Car", color="k")
ax_track.set_xlabel("X [m]", color="k")
ax_track.set_ylabel("Y [m]", color="k")
ax_track.axis("equal")
ax_track.grid(True, color="gray")

# Velocity
vel_line, = ax_vel.plot([], [], "b-")
vel_marker, = ax_vel.plot([], [], "ro")
ax_vel.set_title("Velocity Profile", color="w")
ax_vel.set_xlim(np.min(s), np.max(s))
ax_vel.set_ylim(np.nanmin(v_profile)-1, np.nanmax(v_profile)+1)
ax_vel.grid(True, color="gray")

# Acceleration
acc_line, = ax_acc.plot([], [], "r-")
acc_marker, = ax_acc.plot([], [], "ro")
ax_acc.set_title("Acceleration Profile", color="w")
ax_acc.set_xlim(np.min(s), np.max(s))
ax_acc.set_ylim(np.nanmin(a_profile)-1, np.nanmax(a_profile)+1)
ax_acc.grid(True, color="gray")

# Jerk
jerk_line, = ax_jerk.plot([], [], "g-")
jerk_marker, = ax_jerk.plot([], [], "ro")
ax_jerk.set_title("Jerk Profile", color="w")
ax_jerk.set_xlim(np.min(s), np.max(s))
ax_jerk.set_ylim(np.nanmin(jerk_profile)-1, np.nanmax(jerk_profile)+1)
ax_jerk.grid(True, color="gray")

# Curvature velocity limit
ax_curve.plot(s, v_limit, "c-", linewidth=1.2)
ax_curve.set_title("Vehicle curvature limit", color="w")
ax_curve.grid(True, color="gray")

fig.suptitle("Dynamic Velocity Profiling – Car & Profiles", color="w")

# ==============================
# 7. Animation loop
# ==============================

for i in range(N):
    # Track + car
    car_marker.set_data([x[i]], [y[i]])

    # Velocity
    vel_line.set_data(s[:i+1], v_profile[:i+1])
    vel_marker.set_data([s[i]], [v_profile[i]])

    # Acceleration
    acc_line.set_data(s[:i+1], a_profile[:i+1])
    acc_marker.set_data([s[i]], [a_profile[i]])

    # Jerk
    jerk_line.set_data(s[:i+1], jerk_profile[:i+1])
    jerk_marker.set_data([s[i]], [jerk_profile[i]])

    plt.pause(0.0009)

plt.show()