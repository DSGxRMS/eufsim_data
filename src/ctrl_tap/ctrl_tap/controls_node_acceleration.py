import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
import time

class ControlCircle(Node):
    def __init__(self):
        super().__init__('control_circle', automatically_declare_parameters_from_overrides=True)

        # ---- Params (to tweak at run) ----
        self.declare_parameter('odom_topic', '/ground_truth/odom')
        self.declare_parameter('cmd_topic',  '/cmd')              # change if your sim uses a different cmd topic
        self.declare_parameter('mode',       'ackermann')         # 'ackermann'
        self.declare_parameter('accel',      1.0)                 # m/s^2
        self.declare_parameter('steer',      0.3)                 # rad (≈17 deg), positive = left
        self.declare_parameter('hz',         50.0)                # control rate
        self.declare_parameter('qos_best_effort', True)           # ground truth often BEST_EFFORT

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.cmd_topic  = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.mode       = self.get_parameter('mode').get_parameter_value().string_value.lower()
        self.accel      = float(self.get_parameter('accel').get_parameter_value().double_value)
        self.steer      = float(self.get_parameter('steer').get_parameter_value().double_value)
        self.hz         = float(self.get_parameter('hz').get_parameter_value().double_value)
        self.best_effort= self.get_parameter('qos_best_effort').get_parameter_value().bool_value

        # ---- QoS ----
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if self.best_effort
                        else QoSReliabilityPolicy.RELIABLE
        )

        # ---- Subscriptions ----
        self.last_odom_t = 0.0
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos)

        # ---- Publisher ----
        if self.mode == 'ackermann':
            self.pub_ack = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
            self.pub_twist = None
        else:
            self.pub_twist = self.create_publisher(Twist, self.cmd_topic, 10)
            self.pub_ack = None

        # ---- Loop ----
        self.timer = self.create_timer(1.0 / max(1.0, self.hz), self._tick)

        self.get_logger().info(
            f"[control_circle] odom={self.odom_topic} -> {self.mode}@{self.cmd_topic} "
            f"(accel={self.accel:.2f} m/s^2, steer={self.steer:.3f} rad, "
            f"{'BEST_EFFORT' if self.best_effort else 'RELIABLE'})"
        )

    # ---------- Callbacks ----------
    def _odom_cb(self, _msg: Odometry):
        self.last_odom_t = time.time()

    # ---------- Control loop ----------
    def _tick(self):
        # Minimal “algo”: constant turn + constant acceleration → accelerating circle
        steer_cmd, accel_cmd = self.compute_control_circle()

        if self.pub_ack:
            msg = AckermannDriveStamped()
            msg.drive.steering_angle = steer_cmd
            msg.drive.acceleration   = accel_cmd
            # Many interfaces ignore speed when acceleration is provided; leaving at 0.0 is fine.
            # If your interface *requires* a value, you can set msg.drive.speed to the last odom speed.
            self.pub_ack.publish(msg)
        else:
            # Twist has no acceleration field; keeping API parity only.
            # If your consumer expects Twist, map accel however your stack defines it.
            msg = Twist()
            msg.linear.x = 0.0
            self.pub_twist.publish(msg)

    def compute_control_circle(self):
        """
        Returns (steer [rad], accel [m/s^2]).
        Swap this function later with PPC/Stanley/etc.
        """
        return (self.steer, self.accel)


def main():
    rclpy.init()
    node = ControlCircle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
