#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose2D

# If your gt_pose message is custom (with vx, vy, speed_xy), 
# import that type instead of Pose2D:
# from gt_interfaces.msg import GTPose  # Example

class ControlLoop(Node):
    def __init__(self):
        super().__init__('control_final')

        # ---- Subscribers ----
        self.sub_gt_pose = self.create_subscription(
            Pose2D,          # <-- Change to custom msg if needed
            '/gt_pose',      # <-- Ground truth topic
            self.gt_pose_callback,
            qos_profile_sensor_data
        )

        # ---- Publishers ----
        self.pub_control = self.create_publisher(
            Float32MultiArray,
            '/run_control',  # <-- Control topic
            10
        )

        # ---- State variables ----
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.have_pose = False

        self.timer = self.create_timer(0.05, self.control_step)  # 20 Hz loop
        self.get_logger().info("ControlLoop node started and waiting for /gt_pose...")

    # ------------------------------------------------------------------
    # Receive GT pose updates
    def gt_pose_callback(self, msg):
        self.x = msg.x
        self.y = msg.y
        self.yaw = msg.theta  # For Pose2D
        self.have_pose = True
        # If your message has vx, vy, speed_xy, store them here too:
        # self.vx = msg.vx
        # self.vy = msg.vy
        # self.speed = msg.speed_xy

    # ------------------------------------------------------------------
    # Main control loop
    def control_step(self):
        if not self.have_pose:
            self.get_logger().warn_once("No /gt_pose data received yet.")
            return

        # 🧭 Control logic placeholder:
        # Here, you can implement your own logic for steering/throttle/etc.
        # Example: maintain a constant speed or control orientation.
        desired_yaw = 0.0  # You can set this dynamically later
        yaw_error = desired_yaw - self.yaw
        control_output = Float32MultiArray()
        control_output.data = [yaw_error, self.speed]

        # Publish control command
        self.pub_control.publish(control_output)
        self.get_logger().info(
            f"[Control] x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw:.2f}, control={control_output.data}"
        )

# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ControlLoop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()