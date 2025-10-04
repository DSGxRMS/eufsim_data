#!/usr/bin/env python3
# ppc_node.py — Pure Pursuit Controller for EUFS (Ackermann)
#
# - Loads track from Excel (Skitpad.xlsx by default)
# - Densifies path (B-spline if SciPy available; else linear)
# - Builds curvature-based speed profile (with accel limiting)
# - Curvature-adaptive lookahead, candidate preview scoring
# - Publishes AckermannDriveStamped on /cmd (configurable)
#
# Params (not exhaustive; see __init__):
#   track.file_path (str)         : default "Skitpad.xlsx"
#   track.sheet_name (str|int)    : default 0
#   ctrl.mode ("ackermann")       : default "ackermann"
#   ctrl.cmd_topic (str)          : default "/cmd"
#   ctrl.odom_topic (str)         : default "/ground_truth/odom"
#   ctrl.hz (float)               : default 50.0
#   ctrl.qos_best_effort (bool)   : default True
#   vehicle.wheelbase_m (float)   : default 2.0
#   limits.steer_rad (float)      : default 0.5
#   limits.accel_max (float)      : default 5.0
#   speed.v_cruise (float)        : default 30.0
#   speed.v_min (float)           : default 5.0
#   lookahead.base (float)        : default 5.0
#   lookahead.min (float)         : default 2.0
#   lookahead.max (float)         : default 15.0
#   preview.T (float)             : default 0.5
#   preview.n_candidates (int)    : default 20
#   profile.kappa_speed_scale (float): default 5.0
#
# Notes:
# - If SciPy isn't installed, the node falls back to linear path densification.
# - Requires pandas + openpyxl to read Excel. You can also point file_path to a CSV;
#   we'll try CSV if Excel open fails.
#
import math
import time
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.time import Time as RclTime

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from pathlib import Path
# Optional deps
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

# Try SciPy for splines; fall back to numpy if unavailable
try:
    from scipy.interpolate import splprep, splev, interp1d
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    from numpy import interp as _np_interp  # fallback

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi

def yaw_from_quat(qx, qy, qz, qw) -> float:
    # ZYX yaw
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)



DATADIR = Path(__file__).parent / "aligned_pathpoints.csv"
class PurePursuitNode(Node):
    def __init__(self):
        super().__init__("pure_pursuit_controller", automatically_declare_parameters_from_overrides=True)
        
        # --- timing / controller params (must exist before _build_speed_profile) ---
        self.declare_parameter('ctrl.dt', 0.02)  # 50 Hz default
        # Galactic getter:
        self.dt = float(self.get_parameter('ctrl.dt').get_parameter_value().double_value)

        # If you also use these inside _build_speed_profile, make sure they are set here too:
        self.declare_parameter('ctrl.v_min', 5.0)
        self.declare_parameter('ctrl.v_max', 30.0)
        self.declare_parameter('ctrl.kappa_speed_scale', 5.0)

        self.v_min = float(self.get_parameter('ctrl.v_min').get_parameter_value().double_value)
        self.v_max = float(self.get_parameter('ctrl.v_max').get_parameter_value().double_value)
        self.kappa_speed_scale = float(self.get_parameter('ctrl.kappa_speed_scale').get_parameter_value().double_value)


        # ---- Params (topics / ROS) ----
        
        self.declare_parameter('ctrl.odom_topic', '/ground_truth/odom')
        self.declare_parameter('ctrl.cmd_topic',  '/cmd')
        self.declare_parameter('ctrl.mode',       'ackermann')  # only 'ackermann' supported here
        self.declare_parameter('ctrl.hz',         50.0)
        self.declare_parameter('ctrl.qos_best_effort', True)

        # ---- Path / file params ----
        self.declare_parameter('track.file_path', str(DATADIR))
        self.declare_parameter('track.sheet_name', 0)   # for Excel; ignored for CSV

        # ---- Vehicle + limits ----
        self.declare_parameter('vehicle.wheelbase_m', 2.0)
        self.declare_parameter('limits.steer_rad',    0.5)   # ≈ 28.6°
        self.declare_parameter('limits.accel_max',    5.0)

        # ---- Speed / lookahead / preview params ----
        self.declare_parameter('speed.v_cruise', 30.0)  # m/s
        self.declare_parameter('speed.v_min',     5.0)
        self.declare_parameter('lookahead.base',  5.0)
        self.declare_parameter('lookahead.min',   2.0)
        self.declare_parameter('lookahead.max',   15.0)
        self.declare_parameter('preview.T',       0.5)
        self.declare_parameter('preview.n_candidates', 20)
        self.declare_parameter('profile.kappa_speed_scale', 5.0)

        # ---- Pull params ----
        P = self.get_parameter
        self.odom_topic = str(P('ctrl.odom_topic').value)
        self.cmd_topic  = str(P('ctrl.cmd_topic').value)
        self.mode       = str(P('ctrl.mode').value).lower()
        self.hz         = float(P('ctrl.hz').value)
        # self.hz = 1.0 / max(1e-3, self.dt)
        self.timer = self.create_timer(self.dt, self._tick)

        self.best_effort= bool(P('ctrl.qos_best_effort').value)

        self.file_path  = str(P('track.file_path').value)
        self.sheet_name = P('track.sheet_name').value

        self.L          = float(P('vehicle.wheelbase_m').value)
        self.steer_lim  = float(P('limits.steer_rad').value)
        self.accel_max  = float(P('limits.accel_max').value)

        self.v_cruise   = float(P('speed.v_cruise').value)
        self.v_min      = float(P('speed.v_min').value)
        self.base_look  = float(P('lookahead.base').value)
        self.min_look   = float(P('lookahead.min').value)
        self.max_look   = float(P('lookahead.max').value)
        self.preview_T  = float(P('preview.T').value)
        self.n_candidates = int(P('preview.n_candidates').value)
        self.kappa_speed_scale = float(P('profile.kappa_speed_scale').value)

        # ---- Load path ----
        self._load_path()

        # ---- Build speed profile ----
        self._build_speed_profile()

        # ---- ROS I/O ----
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
            if self.best_effort else QoSReliabilityPolicy.RELIABLE
        )
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos)
        self.pub_ack = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)

        # ---- Control loop ----
        self.dt = 1.0 / max(1.0, self.hz)
        self.timer = self.create_timer(self.dt, self._tick)

        # ---- Runtime state ----
        self.have_odom = False
        self.state = np.array([self.x_dense[0], self.y_dense[0],
                               math.atan2(self.y_dense[1]-self.y_dense[0],
                                          self.x_dense[1]-self.x_dense[0])], float)
        self.last_speed_cmd = 0.0
        self.start_idx = 0
        self.last_tick_wall = time.time()

        self.get_logger().info(
            f"[ppc] up | odom={self.odom_topic} -> ackermann@{self.cmd_topic} | "
            f"rate={self.hz:.1f} Hz | path_pts={len(self.x_dense)} | "
            f"v_cruise={self.v_cruise:.1f} m/s"
        )

    # ---------- Path loading & processing ----------
    def _load_path(self):
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Track file not found: {path}")

        # Load XY columns (Excel preferred; CSV fallback)
        if _HAS_PANDAS:
            df = None
            try:
                # Try Excel first
                import openpyxl  # noqa: F401
                df = pd.read_excel(path, engine="openpyxl", sheet_name=self.sheet_name)
            except Exception:
                try:
                    df = pd.read_csv(path)
                    self.get_logger().warn(f"[ppc] Loaded track as CSV (Excel read failed): {path.name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to read track file: {e}")

            numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if len(numeric_cols) < 2:
                raise ValueError("Track file must contain at least two numeric columns (x, y).")
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            raw_x = df[x_col].to_numpy(dtype=float)
            raw_y = df[y_col].to_numpy(dtype=float)
        else:
            # Extremely minimal CSV loader if pandas not available
            if path.suffix.lower() != ".csv":
                raise RuntimeError("pandas not installed. Use CSV file or install pandas+openpyxl.")
            raw = np.loadtxt(path, delimiter=",", ndmin=2)
            if raw.shape[1] < 2:
                raise ValueError("CSV must have at least 2 columns (x, y).")
            raw_x, raw_y = raw[:,0], raw[:,1]

        # Build cumulative arclength on raw points (for parameterization)
        dx = np.diff(raw_x); dy = np.diff(raw_y)
        seglen = np.hypot(dx, dy)
        if np.all(seglen < 1e-9):
            raise ValueError("Track points are all identical or too close.")
        s = np.concatenate([[0.0], np.cumsum(seglen)])
        t_raw = s / s[-1]

        # Densify with B-spline if available, else linear
        try:
            if _HAS_SCIPY:
                tck, _ = splprep([raw_x, raw_y], u=t_raw, s=0.0, k=min(3, len(raw_x)-1))
                u_dense = np.linspace(0.0, 1.0, max(1000, len(raw_x)*5))
                x_dense, y_dense = splev(u_dense, tck)
                dx1, dy1 = splev(u_dense, tck, der=1)
                dx2, dy2 = splev(u_dense, tck, der=2)
                denom = (np.hypot(dx1, dy1)**3) + 1e-9
                kappa = (dx1*dy2 - dy1*dx2) / denom
            else:
                raise RuntimeError("No SciPy")
        except Exception:
            # Linear interpolation fallback
            u_dense = np.linspace(0.0, 1.0, max(2000, len(raw_x)*5))
            x_dense = np.interp(u_dense, t_raw, raw_x)
            y_dense = np.interp(u_dense, t_raw, raw_y)
            dx1 = np.gradient(x_dense); dy1 = np.gradient(y_dense)
            dx2 = np.gradient(dx1);     dy2 = np.gradient(dy1)
            denom = (np.hypot(dx1, dy1)**3) + 1e-9
            kappa = (dx1*dy2 - dy1*dx2) / denom

        abs_kappa = np.abs(kappa)
        s_dense = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x_dense), np.diff(y_dense)))])

        self.x_dense = np.asarray(x_dense, float)
        self.y_dense = np.asarray(y_dense, float)
        self.abs_kappa = np.asarray(abs_kappa, float)
        self.s_dense = np.asarray(s_dense, float)

    def _build_speed_profile(self):
        v_min = self.v_min
        v_max = self.v_cruise
        ks = self.kappa_speed_scale

        v_profile = v_max / (1.0 + ks*self.abs_kappa)
        v_profile = np.clip(v_profile, v_min, v_max)

        # Smooth (≈ 0.5 s box)
        win = max(3, int(0.5 / max(1e-3, self.dt)))
        box = np.ones(win, float) / win
        v_smooth = np.convolve(v_profile, box, mode='same')

        # Interp speed as a function of arclength
        if _HAS_SCIPY:
            self.v_along_s = interp1d(self.s_dense, v_smooth,
                                      bounds_error=False,
                                      fill_value=(float(v_smooth[0]), float(v_smooth[-1])))
        else:
            # numpy fallback
            def _v_s(s_query: float) -> float:
                s = self.s_dense
                v = v_smooth
                if s_query <= s[0]:
                    return float(v[0])
                if s_query >= s[-1]:
                    return float(v[-1])
                idx = np.searchsorted(s, s_query)
                s0, s1 = s[idx-1], s[idx]
                v0, v1 = v[idx-1], v[idx]
                a = (s_query - s0) / max(1e-9, (s1 - s0))
                return float((1.0 - a)*v0 + a*v1)
            self.v_along_s = _v_s

    # ---------- ROS Callbacks ----------
    def _odom_cb(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.state = np.array([x, y, yaw], float)
        self.have_odom = True

    # ---------- Helpers ----------
    def _closest_index_on_path(self, px: float, py: float, start_idx: int = 0):
        # Linear scan forward from last index (track is dense; this is OK)
        xd = self.x_dense[start_idx:]
        yd = self.y_dense[start_idx:]
        d2 = (xd - px)*(xd - px) + (yd - py)*(yd - py)
        i_rel = int(np.argmin(d2))
        idx = start_idx + i_rel
        return idx, float(math.sqrt(d2[i_rel]))

    def _curvature_adaptive_lookahead(self, k_local: float) -> float:
        look = self.base_look / (1.0 + 10.0 * k_local)
        return float(np.clip(look, self.min_look, self.max_look))

    def _pure_pursuit_delta(self, state, start_idx: int, lookahead_m: float):
        x, y, yaw = state
        idx = start_idx
        # march forward until we accumulated ~lookahead_m along the path
        while idx < len(self.x_dense) - 1:
            dx = self.x_dense[idx+1] - x
            dy = self.y_dense[idx+1] - y
            d = math.hypot(dx, dy)
            if d >= lookahead_m:
                idx += 1
                break
            idx += 1
        idx = min(idx, len(self.x_dense) - 1)
        gx = self.x_dense[idx]; gy = self.y_dense[idx]
        alpha = wrap_angle(math.atan2(gy - y, gx - x) - yaw)
        Ld = max(1e-3, math.hypot(gx - x, gy - y))
        delta = math.atan2(2.0 * self.L * math.sin(alpha), Ld)
        return float(np.clip(delta, -self.steer_lim, self.steer_lim)), idx

    def _step_kinematic(self, state, delta: float, v: float, dt: float):
        x, y, yaw = state
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = wrap_angle(yaw + (v / self.L) * math.tan(delta) * dt)
        return np.array([x, y, yaw], float)

    def _score_candidate(self, state, start_idx: int, lookahead: float, v_local: float, steps: int):
        sim_state = state.copy()
        idx = start_idx
        cte_total = 0.0
        for _ in range(steps):
            delta, idx = self._pure_pursuit_delta(sim_state, idx, lookahead)
            sim_state = self._step_kinematic(sim_state, delta, v_local, self.dt)
            # CTE: distance to nearest future point
            i_closest, d_closest = self._closest_index_on_path(sim_state[0], sim_state[1], start_idx)
            cte_total += d_closest
        return cte_total

    # ---------- Control Loop ----------
    def _tick(self):
        if not self.have_odom:
            return

        # Estimated current path index
        self.start_idx, _ = self._closest_index_on_path(self.state[0], self.state[1], self.start_idx)

        # Local curvature window
        k_local = float(np.max(self.abs_kappa[self.start_idx: min(self.start_idx + 10, len(self.abs_kappa))]))
        baseline_look = self._curvature_adaptive_lookahead(k_local)

        # Speed command from arclength profile (with accel limiting)
        # Approximate "s here" by path index arclength
        s_here = float(self.s_dense[self.start_idx])
        v_ref = float(self.v_along_s(s_here)) if callable(self.v_along_s) else float(self.v_along_s)
        v_cmd = float(np.clip(v_ref,
                              self.last_speed_cmd - self.accel_max * self.dt,
                              self.last_speed_cmd + self.accel_max * self.dt))
        v_cmd = float(np.clip(v_cmd, self.v_min, self.v_cruise))

        # Candidate lookahead sampling around predicted preview distance
        n_cand = max(1, self.n_candidates)
        best_score = float('inf'); best_look = baseline_look
        preview_look = np.clip(v_cmd * self.preview_T, self.min_look, self.max_look)
        for _ in range(n_cand):
            look = 0.6 * preview_look + 0.4 * baseline_look
            score = self._score_candidate(self.state, self.start_idx, look, v_cmd, steps=max(1, int(1.0 / self.dt)))
            if score < best_score:
                best_score = score
                best_look = look

        # Final steering via pure pursuit
        delta, _ = self._pure_pursuit_delta(self.state, self.start_idx, best_look)

        # Publish Ackermann (EUFS usually honors speed directly; acceleration optional)
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(delta)
        msg.drive.speed = float(v_cmd)            # main speed command
        msg.drive.acceleration = float(self.accel_max)  # optional; many stacks ignore if speed set
        self.pub_ack.publish(msg)

        self.last_speed_cmd = v_cmd

        # Optional light log (1 Hz)
        now = time.time()
        if now - getattr(self, "_last_log", 0.0) > 1.0:
            self._last_log = now
            self.get_logger().info(
                f"[ppc] idx={self.start_idx} v={v_cmd:4.1f} m/s  steer={delta: .3f} rad  look={best_look:4.1f} m  κloc={k_local:.3f}"
            )

def main():
    rclpy.init()
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
