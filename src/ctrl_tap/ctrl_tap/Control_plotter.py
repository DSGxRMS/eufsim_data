#!/usr/bin/env python3
# control_plotter.py
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose2D
import matplotlib
matplotlib.use("TkAgg")  # or change to Agg if headless
import matplotlib.pyplot as plt
import math
import time

class ControlPlotter(Node):
    def __init__(self):
        super().__init__('control_plotter')

        # ---- Subscribers ----
        self.sub_control = self.create_subscription(
            Float32MultiArray, '/run_control', self.control_callback, qos_profile_sensor_data
        )
        self.sub_gt_pose = self.create_subscription(
            Pose2D, '/gt_pose', self.pose_callback, qos_profile_sensor_data
        )

        # ---- Data storage ----
        self.distances = [0.0]
        self.speeds = []
        self.steering = []
        self.last_pose = None
        self.total_distance = 0.0

        # ---- Plot setup ----
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.line_speed, = self.ax1.plot([], [], label='Speed (m/s)')
        self.line_steer, = self.ax2.plot([], [], label='Steering (yaw_error)')
        self.ax1.set_ylabel("Speed (m/s)")
        self.ax2.set_ylabel("Steering")
        self.ax2.set_xlabel("Distance (m)")
        self.ax1.grid(True, alpha=0.3)
        self.ax2.grid(True, alpha=0.3)
        self.ax1.legend()
        self.ax2.legend()

        # Timer for refreshing plot every 0.5s
        self.last_update_time = time.time()
        self.update_interval = 0.5

        self.get_logger().info("Live ControlPlotter started — plotting /run_control vs. distance in real-time")

    # ------------------------------------------------------------------
    def pose_callback(self, msg: Pose2D):
        x, y = msg.x, msg.y
        if self.last_pose is not None:
            dx = x - self.last_pose[0]
            dy = y - self.last_pose[1]
            d = math.hypot(dx, dy)
            if d > 0.0:
                self.total_distance += d
                self.distances.append(self.total_distance)
        self.last_pose = (x, y)

    # ------------------------------------------------------------------
    def control_callback(self, msg: Float32MultiArray):
        if len(msg.data) >= 2:
            yaw_error, speed = msg.data[0], msg.data[1]
            self.steering.append(float(yaw_error))
            self.speeds.append(float(speed))

        # Update plot occasionally
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            self.update_plot()
            self.last_update_time = current_time

    # ------------------------------------------------------------------
    def update_plot(self):
        if len(self.distances) < 2:
            return

        n = min(len(self.distances), len(self.speeds))
        if n > 1:
            self.line_speed.set_data(self.distances[:n], self.speeds[:n])
            self.ax1.relim()
            self.ax1.autoscale_view()

        n2 = min(len(self.distances), len(self.steering))
        if n2 > 1:
            self.line_steer.set_data(self.distances[:n2], self.steering[:n2])
            self.ax2.relim()
            self.ax2.autoscale_view()

        plt.pause(0.001)

    # ------------------------------------------------------------------
    def on_shutdown(self):
        self.get_logger().info("Shutting down plotter…")
        plt.ioff()
        try:
            plt.show()
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = ControlPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.on_shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
