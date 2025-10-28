import math
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from scipy.interpolate import splprep, splev

# -------------------- Parameters --------------------
V_MAX = 8.0
V_MIN = 5.0
A_MAX = 15.0
D_MAX = 20.0
J_MAX = 70.0
CURVATURE_MAX = 0.8
WHEELBASE_M = 1.5


class VelocityProfiler(Node):
    def __init__(self):
        super().__init__('velocity_profiler', automatically_declare_parameters_from_overrides=True)

        # Declare topics
        self.declare_parameter('path_topic', '/path_points')
        self.declare_parameter('profile_topic', '/velocity_profile')

        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        self.profile_topic = self.get_parameter('profile_topic').get_parameter_value().string_value

        # Subscriber & Publisher
        self.create_subscription(Path, self.path_topic, self.path_callback, 10)
        self.profile_pub = self.create_publisher(Float32MultiArray, self.profile_topic, 10)

        self.get_logger().info(f"[velocity_profiler] Listening to {self.path_topic}, publishing to {self.profile_topic}")

        # Storage
        self.last_profile = None

    # -------------------- Path Callback --------------------
    def path_callback(self, msg: Path):
        if not msg.poses:
            return

        xs = np.array([p.pose.position.x for p in msg.poses])
        ys = np.array([p.pose.position.y for p in msg.poses])

        xs, ys = self.resample_track(xs, ys)
        kappa = self.compute_signed_curvature(xs, ys)
        v_limit = self.ackermann_curv_speed_limit(kappa)
        ds = self.segment_distances(xs, ys)
        v_profile = self.jerk_limited_velocity_profile(v_limit, ds, 0.0, 0.0,
                                                       V_MIN, V_MAX, A_MAX, D_MAX, J_MAX)

        # Publish live profile
        msg_out = Float32MultiArray()
        msg_out.data = v_profile.tolist()
        self.profile_pub.publish(msg_out)

        self.get_logger().info(f"Published velocity profile ({len(v_profile)} points)")
        self.last_profile = v_profile

    # -------------------- Helper Functions --------------------
    def resample_track(self, x_raw, y_raw, num_points=800):
        tck, _ = splprep([x_raw, y_raw], s=0, k=min(3, max(1, len(x_raw) - 1)))
        tt_dense = np.linspace(0, 1, 2000)
        xx, yy = splev(tt_dense, tck)
        s_dense = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))))
        s_dense /= s_dense[-1] if s_dense[-1] > 0 else 1.0
        s_uniform = np.linspace(0, 1, num_points)
        return np.interp(s_uniform, s_dense, xx), np.interp(s_uniform, s_dense, yy)

    def compute_signed_curvature(self, x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = np.power(dx * dx + dy * dy, 1.5) + 1e-12
        kappa = (dx * ddy - dy * ddx) / denom
        kappa = np.clip(kappa, -CURVATURE_MAX, CURVATURE_MAX)
        kappa[~np.isfinite(kappa)] = 0.0
        return kappa

    def ackermann_curv_speed_limit(self, kappa):
        delta = np.arctan(kappa * WHEELBASE_M)
        denom = np.abs(np.tan(delta)) + 1e-6
        v = np.sqrt(np.maximum(0.0, (D_MAX * WHEELBASE_M) / denom))
        return np.minimum(v, V_MAX)

    def segment_distances(self, xs, ys):
        x_next = np.roll(xs, -1)
        y_next = np.roll(ys, -1)
        ds = np.hypot(x_next - xs, y_next - ys)
        ds[-1] = 0.0
        return ds

    def jerk_limited_velocity_profile(self, v_limit, ds, v0, vf, v_min, v_max, a_max, d_max, j_max):
        v_limit = np.asarray(v_limit, dtype=float)
        ds = np.asarray(ds, dtype=float)
        N = len(v_limit)

        v_forward = np.zeros(N)
        v_forward[0] = min(max(v0, 0.0), v_limit[0], v_max)
        a_prev = 0.0

        # Forward pass
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

        # Backward pass
        a_prev = 0.0
        for i in range(N - 2, -1, -1):
            ds_i = max(ds[i + 1], 1e-9)
            v_avg = max(v_min, v_profile[i + 1])
            dt = ds_i / v_avg
            a_curr = min(a_prev + j_max * dt, d_max)
            v_possible = math.sqrt(max(0.0, v_profile[i + 1] ** 2 + 2.0 * a_curr * ds_i))
            v_profile[i] = min(v_profile[i], v_possible, v_max)
            a_prev = (v_profile[i + 1] ** 2 - v_profile[i] ** 2) / (2.0 * ds_i)

        v_profile = np.clip(v_profile, v_min, v_max)
        if vf <= 0.0:
            v_profile[-1] = 0.0
        return v_profile


def main(args=None):
    rclpy.init(args=args)
    node = VelocityProfiler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()