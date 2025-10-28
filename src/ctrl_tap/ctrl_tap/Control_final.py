#!/usr/bin/env python3
# ppc_controller_refit.py
#
# Refit of your controller to match the SIM I/O template:
#  - Subscribes Odometry from /ground_truth/odom (QoS BEST_EFFORT by default)
#  - Publishes AckermannDriveStamped on /cmd (or Twist if mode != 'ackermann')
#  - Preserves your planning/control logic and parameters
#  - Adds optional telemetry topics for your plotter: /run_control (Float32MultiArray), /gt_pose (Pose2D)

import rclpy
import threading, time, math
import numpy as np
import pandas as pd

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray

from control.control_utils import *  # your helpers (resample_track, preprocess_path, etc.)
from pathlib import Path

# -------------------- Path / scenario constants --------------------
PATHPOINTS_CSV = Path(__file__).parent / "pathpoints.csv"
ROUTE_IS_LOOP = False
scaling_factor = 1.0

# -------------------- Controller / vehicle params --------------------
WHEELBASE_M = 1.5
MAX_STEER_RAD = 1.0  # ~57 deg (you had "10 deg" in comment; keep 1.0 rad as in code)
PROFILE_WINDOW_M = 100.0
PROFILE_HZ = 20
BRAKE_GAIN = 0.7
STOP_SPEED_THRESHOLD = 0.1   # m/s

# -------------------- Jerk-limited velocity profile params --------------------
V_MIN = 4.0
V_MAX = 8.0
A_MAX = 15.0
D_MAX = 20.0
J_MAX = 70.0

# -------------------- Pure pursuit velocity limiting --------------------
STEER_SPEED_LIMIT_FACTOR   = 0.5   # factor to reduce speed based on steering angle
MAX_STEER_FOR_SPEED_LIMIT  = 0.3   # radians where speed limiting starts
STEER_RATE_THRESHOLD       = 0.5   # rad/s where additional speed limiting starts
STEER_RATE_FACTOR          = 0.2   # how much to reduce speed based on steering rate

# -------------------- Multi-point curvature parameters --------------------
CURVATURE_SAMPLE_POINTS    = 8
CURVATURE_LOOKAHEAD_FACTOR = 0.3
CURVATURE_MAX              = 0.20  # clamp for pp curvature heuristic (≈ 5 m min radius)

def quat_to_yaw(qx, qy, qz, qw):
    # yaw from quaternion (ZYX convention)
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

class PathFollower(Node):
    def __init__(self):
        super().__init__('ppc_controller', automatically_declare_parameters_from_overrides=True)

        # ---- Runtime I/O params, aligned with your SIM template ----
        self.declare_parameter('odom_topic', '/ground_truth/odom')
        self.declare_parameter('cmd_topic',  '/cmd')
        self.declare_parameter('mode',       'ackermann')     # 'ackermann' (default)
        self.declare_parameter('hz',         50.0)
        self.declare_parameter('qos_best_effort', True)

        self.odom_topic  = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.cmd_topic   = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.mode        = self.get_parameter('mode').get_parameter_value().string_value.lower()
        self.ctrl_hz     = float(self.get_parameter('hz').get_parameter_value().double_value)
        self.best_effort = self.get_parameter('qos_best_effort').get_parameter_value().bool_value

        # ---- QoS ----
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if self.best_effort else QoSReliabilityPolicy.RELIABLE
        )

        # ---- Subscriptions ----
        self._have_odom = False
        self._cx = self._cy = 0.0
        self._yaw = 0.0
        self._speed = 0.0
        self._last_odom_t = 0.0

        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos)

        # ---- Publishers ----
        if self.mode == 'ackermann':
            self.pub_ack = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
            self.pub_twist = None
        else:
            self.pub_twist = self.create_publisher(Twist, self.cmd_topic, 10)
            self.pub_ack = None

        # Optional telemetry for your plotter
        self.pub_plot = self.create_publisher(Float32MultiArray, '/run_control', 10)
        self.pub_gt   = self.create_publisher(Pose2D, '/gt_pose', 10)

        # ---- Load and preprocess path ----
        df = pd.read_csv(PATHPOINTS_CSV)
        rx, ry = resample_track(df["x"].to_numpy() * scaling_factor,
                                df["y"].to_numpy() * scaling_factor)

        # Your original axis swap/offset
        route_x, route_y = ry + 15.0, -rx
        route_x, route_y, route_s, route_len = preprocess_path(route_x, route_y, loop=ROUTE_IS_LOOP)
        seg_ds = segment_distances(route_x, route_y, ROUTE_IS_LOOP)

        kappa_signed = compute_signed_curvature(route_x, route_y)
        v_limit_global = ackermann_curv_speed_limit(kappa_signed, wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX)
        route_v = v_limit_global.copy()

        # ---- Controllers ----
        th_pid = PID(3.2, 0.0, 0.0)
        steer_pid = PIDRange(kp=0.15, ki=0.05, kd=0.20, out_min=-0.60, out_max=0.60)

        # ---- Loop state ----
        cur_idx = 0
        last_t = time.perf_counter()
        last_profile_t = last_t
        prev_steering = 0.0  # rad

        self.get_logger().info(
            f"[ppc_controller] odom={self.odom_topic} -> {self.mode}@{self.cmd_topic} "
            f"(QoS={'BEST_EFFORT' if self.best_effort else 'RELIABLE'})"
        )

        # ---- Timer loop ----
        period = 1.0 / max(1.0, self.ctrl_hz)
        self._timer = self.create_timer(period, lambda: self._tick(
            route_x, route_y, route_s, route_len, seg_ds,
            kappa_signed, v_limit_global, route_v,
            th_pid, steer_pid,
            context={'cur_idx': cur_idx,
                     'last_t': last_t,
                     'last_profile_t': last_profile_t,
                     'prev_steering': prev_steering}
        ))

    # ---------- Odometry callback ----------
    def _odom_cb(self, msg: Odometry):
        self._cx = msg.pose.pose.position.x
        self._cy = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._speed = math.hypot(vx, vy)

        self._have_odom = True
        self._last_odom_t = time.time()

        # Publish GT pose for plotter
        pose2d = Pose2D()
        pose2d.x = self._cx
        pose2d.y = self._cy
        pose2d.theta = self._yaw
        self.pub_gt.publish(pose2d)

    # ---------- Main control tick ----------
    def _tick(self, route_x, route_y, route_s, route_len, seg_ds,
              kappa_signed, v_limit_global, route_v,
              th_pid, steer_pid, context):

        if not self._have_odom:
            return

        # Unpack local state
        cur_idx        = context['cur_idx']
        last_t         = context['last_t']
        last_profile_t = context['last_profile_t']
        prev_steering  = context['prev_steering']

        cx, cy, yaw, speed = self._cx, self._cy, self._yaw, self._speed

        now = time.perf_counter()
        dt = max(1e-3, now - last_t)  # guard
        last_t = now

        # update index
        cur_idx = local_closest_index((cx, cy), route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)

        # refresh velocity profile (periodically)
        if now - last_profile_t >= (1.0 / PROFILE_HZ):
            last_profile_t = now

            # lookahead & pp curvature heuristic
            Ld = calc_lookahead(speed)
            tgt_idx = forward_index_by_distance(cur_idx, Ld, route_s, route_len, ROUTE_IS_LOOP)

            dx = route_x[tgt_idx] - cx
            dy = route_y[tgt_idx] - cy
            look_ahead_distance = math.hypot(dx, dy)

            e_lat, _ = cross_track_error(cx, cy, route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
            path_yaw = path_heading(route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)

            heading_error = yaw - path_yaw
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

            if look_ahead_distance > 0.1:
                pp_curvature = 2.0 * e_lat / (look_ahead_distance ** 2)
                pp_curvature = max(-CURVATURE_MAX, min(CURVATURE_MAX, pp_curvature))
            else:
                pp_curvature = 0.0

            # --- Multi-point conservative limit ---
            curvature_sample_points = []
            curvature_lookahead = min(PROFILE_WINDOW_M * CURVATURE_LOOKAHEAD_FACTOR, PROFILE_WINDOW_M)
            for i in range(CURVATURE_SAMPLE_POINTS):
                frac = i / max(1, (CURVATURE_SAMPLE_POINTS - 1))
                sample_dist = frac * curvature_lookahead
                sample_idx = forward_index_by_distance(cur_idx, sample_dist, route_s, route_len, ROUTE_IS_LOOP)
                curvature_sample_points.append(sample_idx)

            conservative_speed_limit = V_MAX
            for idx in curvature_sample_points:
                if idx != cur_idx:
                    e_lat_sample, _ = cross_track_error(route_x[idx], route_y[idx], route_x, route_y,
                                                        cur_idx, loop=ROUTE_IS_LOOP)
                    sample_lookahead = calc_lookahead(speed)
                    if sample_lookahead > 0.1:
                        pp_curv_sample = 2.0 * e_lat_sample / (sample_lookahead ** 2)
                        pp_curv_sample = max(-CURVATURE_MAX, min(CURVATURE_MAX, pp_curv_sample))
                    else:
                        pp_curv_sample = 0.0

                    pp_safe_speed = ackermann_curv_speed_limit(np.array([pp_curv_sample]),
                                                               wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX)[0]

                    path_curvature = kappa_signed[idx]
                    path_safe_speed = ackermann_curv_speed_limit(np.array([path_curvature]),
                                                                 wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX)[0]

                    conservative_speed_limit = min(conservative_speed_limit,
                                                   min(pp_safe_speed, path_safe_speed))

            # window idxs
            end_idx = forward_index_by_distance(cur_idx, PROFILE_WINDOW_M, route_s, route_len, ROUTE_IS_LOOP)
            if ROUTE_IS_LOOP:
                if end_idx >= cur_idx:
                    idxs = np.arange(cur_idx, end_idx + 1)
                else:
                    idxs = np.concatenate((np.arange(cur_idx, len(route_x)), np.arange(0, end_idx + 1)))
            else:
                idxs = np.arange(cur_idx, end_idx + 1)

            # ds window
            if ROUTE_IS_LOOP:
                ds_win_list = [0.0]
                for ii in range(len(idxs) - 1):
                    ds_win_list.append(seg_ds[idxs[ii]])
                ds_win = np.asarray(ds_win_list, dtype=float)
            else:
                ds_win = np.concatenate(([0.0], seg_ds[idxs[:-1]])) if len(idxs) > 0 else np.asarray([0.0])

            # apply limit + jerk profile
            v_lim_win = np.minimum(v_limit_global[idxs], conservative_speed_limit)
            v0 = speed
            vf = 0.0 if (not ROUTE_IS_LOOP and len(idxs) > 0 and idxs[-1] == len(route_x) - 1) else float(v_lim_win[-1])

            v_prof_win = jerk_limited_velocity_profile(
                v_lim_win, ds_win, v0, vf, V_MIN, V_MAX, A_MAX, D_MAX, J_MAX
            )

            if ROUTE_IS_LOOP and end_idx < cur_idx:
                n1 = len(route_x) - cur_idx
                route_v[cur_idx:] = v_prof_win[:n1]
                route_v[:end_idx + 1] = v_prof_win[n1:]
            else:
                route_v[cur_idx:end_idx + 1] = v_prof_win

        # --- Steering control ---
        steering_ppc, tgt_idx = pure_pursuit_steer((cx, cy), yaw, speed,
                                                   route_x, route_y, cur_idx,
                                                   route_s, route_len, loop=ROUTE_IS_LOOP)

        e_lat, yaw_err = cross_track_error(cx, cy, route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
        steer_correction = steer_pid.update(e_lat, dt)
        steering_cmd = max(-1.0, min(1.0, steering_ppc + steer_correction))  # normalized [-1,1]

        # --- Steering rate consideration ---
        steering_rate = abs(steering_cmd * MAX_STEER_RAD - prev_steering) / dt
        prev_steering = steering_cmd * MAX_STEER_RAD

        # --- Angle + rate speed limiting ---
        steering_rate_factor = 1.0
        if steering_rate > STEER_RATE_THRESHOLD:
            rate_reduction = (steering_rate - STEER_RATE_THRESHOLD) * STEER_RATE_FACTOR
            steering_rate_factor = max(0.1, 1.0 - rate_reduction)

        abs_steering = abs(steering_cmd * MAX_STEER_RAD)
        if abs_steering > MAX_STEER_FOR_SPEED_LIMIT:
            steering_factor = max(0.1, 1.0 - (abs_steering - MAX_STEER_FOR_SPEED_LIMIT) * STEER_SPEED_LIMIT_FACTOR)
            angle_factor = steering_factor
        else:
            angle_factor = 1.0

        speed_limit_factor = min(steering_rate_factor, angle_factor)
        limited_speed = route_v[cur_idx] * speed_limit_factor

        # --- Longitudinal control ---
        v_err = limited_speed - speed
        if v_err >= 0:
            throttle = th_pid.update(v_err, dt)
            brake = 0.0
        else:
            th_pid.reset()
            throttle, brake = 0.0, min(1.0, -v_err * BRAKE_GAIN)

        # Clamp [0,1]
        throttle = max(0.0, min(1.0, throttle))
        brake    = max(0.0, min(1.0, brake))

        # --- Publish command (SIM template) ---
        if self.pub_ack:
            msg = AckermannDriveStamped()
            msg.drive.steering_angle = steering_cmd * MAX_STEER_RAD
            # NOTE: Many sims ignore speed when accel provided; adjust if your bridge expects speed.
            msg.drive.speed         = float(limited_speed)
            msg.drive.acceleration  = float(throttle - brake)  # your interface may map this differently
            self.pub_ack.publish(msg)
        else:
            # parity for Twist mode (no acceleration field)
            msg = Twist()
            msg.linear.x  = float(limited_speed)
            msg.angular.z = float(steering_cmd * MAX_STEER_RAD)  # if your stack uses yaw-rate instead, change mapping
            self.pub_twist.publish(msg)

        # --- Telemetry for plotter (/run_control: [yaw_error, speed]) ---
        tele = Float32MultiArray()
        tele.data = [float(yaw_err), float(speed)]
        self.pub_plot.publish(tele)

        # --- Exit condition for non-loop routes ---
        if (not ROUTE_IS_LOOP) and cur_idx >= len(route_x) - 1 and speed < STOP_SPEED_THRESHOLD:
            self.get_logger().info("Reached end of route and stopped. Exiting.")
            rclpy.shutdown()
            return

        # Persist updated locals
        context['cur_idx']        = cur_idx
        context['last_t']         = last_t
        context['last_profile_t'] = last_profile_t
        context['prev_steering']  = prev_steering

def main():
    rclpy.init()
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
