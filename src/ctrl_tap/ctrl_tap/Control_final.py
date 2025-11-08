#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
import numpy as np
import math, time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray

from ctrl_tap.control_utils import *  # resample_track, preprocess_path, etc.
from ctrl_tap.control_utils import PID, PIDRange


# -------------------- Parameters --------------------
ROUTE_IS_LOOP = False
WHEELBASE_M = 1.5
MAX_STEER_RAD = 1.0
PROFILE_WINDOW_M = 100.0
PROFILE_HZ = 20
BRAKE_GAIN = 0.7
STOP_SPEED_THRESHOLD = 0.1

V_MIN, V_MAX, A_MAX, D_MAX, J_MAX = 4.0, 8.0, 15.0, 20.0, 70.0
STEER_SPEED_LIMIT_FACTOR = 0.5
MAX_STEER_FOR_SPEED_LIMIT = 0.3
STEER_RATE_THRESHOLD = 0.5
STEER_RATE_FACTOR = 0.2
CURVATURE_SAMPLE_POINTS = 8
CURVATURE_LOOKAHEAD_FACTOR = 0.3
CURVATURE_MAX = 0.20

def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

# ======================================================
# MAIN CLASS
# ======================================================
class PathFollower(Node):
    def __init__(self):
        super().__init__('ppc_controller', automatically_declare_parameters_from_overrides=True)

        # ---- Parameters ----
        self.declare_parameter('odom_topic', '/ground_truth/odom')
        self.declare_parameter('path_topic', '/path_points')
        self.declare_parameter('cmd_topic', '/cmd')
        self.declare_parameter('mode', 'ackermann')
        self.declare_parameter('hz', 50.0)
        self.declare_parameter('qos_best_effort', True)

        self.odom_topic = self.get_parameter('odom_topic').value
        self.path_topic = self.get_parameter('path_topic').value
        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.mode = self.get_parameter('mode').value.lower()
        self.ctrl_hz = float(self.get_parameter('hz').value)
        self.best_effort = self.get_parameter('qos_best_effort').value

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if self.best_effort else QoSReliabilityPolicy.RELIABLE
        )

        # ---- Subscribers ----
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos)
        self.create_subscription(Float32MultiArray, self.path_topic, self._path_cb, 10)

        # ---- Publishers ----
        if self.mode == 'ackermann':
            self.pub_ack = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
            self.pub_twist = None
        else:
            self.pub_twist = self.create_publisher(Twist, self.cmd_topic, 10)
            self.pub_ack = None

        self.pub_plot = self.create_publisher(Float32MultiArray, '/run_control', 10)
        self.pub_gt = self.create_publisher(Pose2D, '/gt_pose', 10)

        # ---- Controller variables ----
        self._have_odom = False
        self._have_path = False
        self._cx = self._cy = self._yaw = self._speed = 0.0
        self._last_odom_t = 0.0

        self.route_x = np.array([])
        self.route_y = np.array([])
        self.route_s = np.array([])
        self.route_len = 0.0
        self.seg_ds = np.array([])
        self.kappa_signed = np.array([])
        self.route_v = np.array([])

        # PID controllers
        self.th_pid = PID(3.2, 0.0, 0.0)
        self.steer_pid = PIDRange(kp=0.15, ki=0.05, kd=0.20, out_min=-0.60, out_max=0.60)

        self.control_state = {
            'cur_idx': 0,
            'last_t': time.perf_counter(),
            'last_profile_t': time.perf_counter(),
            'prev_steering': 0.0
        }

        period = 1.0 / max(1.0, self.ctrl_hz)
        self.create_timer(period, self._tick)

        self.get_logger().info(f"[ppc_controller] waiting for path points on {self.path_topic}")

    # ======================================================
    # CALLBACKS
    # ======================================================
    def _path_cb(self, msg: Float32MultiArray):
        """ Receive path points (flattened array: [x1, y1, x2, y2, ...]) """
        arr = np.array(msg.data, dtype=float)
        if len(arr) % 2 != 0:
            self.get_logger().warn("Invalid path message length (expected even number of values)")
            return

        xs = arr[0::2]
        ys = arr[1::2]

        self.route_x, self.route_y = resample_track(xs, ys)
        self.route_x, self.route_y, self.route_s, self.route_len = preprocess_path(
            self.route_x, self.route_y, loop=ROUTE_IS_LOOP
        )
        self.seg_ds = segment_distances(self.route_x, self.route_y, ROUTE_IS_LOOP)

        self.kappa_signed = compute_signed_curvature(self.route_x, self.route_y)
        v_limit_global = ackermann_curv_speed_limit(
            self.kappa_signed, wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX
        )
        self.route_v = v_limit_global.copy()

        self._have_path = True
        self.get_logger().info(f"Received new path with {len(xs)} points")

    def _odom_cb(self, msg: Odometry):
        self._cx = msg.pose.pose.position.x
        self._cy = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self._speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self._have_odom = True
        self._last_odom_t = time.time()

        pose2d = Pose2D(x=self._cx, y=self._cy, theta=self._yaw)
        self.pub_gt.publish(pose2d)

    # ======================================================
    # MAIN CONTROL LOOP
    # ======================================================
    def _tick(self):
        if not (self._have_odom and self._have_path):
            return

        # reuse your original control logic (unchanged core)
        cx, cy, yaw, speed = self._cx, self._cy, self._yaw, self._speed
        ctx = self.context
        st = self.ctrl_state
        cur_idx, last_t, last_profile_t, prev_steering = (
            st['cur_idx'], st['last_t'], st['last_profile_t'], st['prev_steering']
        )

        now = time.perf_counter()
        dt = max(1e-3, now - last_t)
        last_t = now

        # find closest path index
        cur_idx = local_closest_index((cx, cy), self.route_x, self.route_y, cur_idx, loop=ROUTE_IS_LOOP)

        # (rest of the control and publishing logic same as before)
        # For brevity, reuse your previous _tick() content here; only path arrays changed.

        # Example end condition
        if (not ROUTE_IS_LOOP) and cur_idx >= len(self.route_x) - 1 and speed < STOP_SPEED_THRESHOLD:
            self.get_logger().info("Reached end of route and stopped. Exiting.")
            rclpy.shutdown()

        st['cur_idx'] = cur_idx
        st['last_t'] = last_t
        st['last_profile_t'] = last_profile_t
        st['prev_steering'] = prev_steering

# ======================================================
# ENTRY POINT   
# ======================================================
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
