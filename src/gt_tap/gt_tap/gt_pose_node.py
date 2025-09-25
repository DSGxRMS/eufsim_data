#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from gt_tap.gt_utils import ensure_dir, quat_to_yaw, norm2

class GTPoseTap(Node):
    def __init__(self):
        super().__init__('gt_pose_tap')
        topic = '/ground_truth/odom'
        self.sub = self.create_subscription(Odometry, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing: {topic}")

      
        
    def _home(self):
        import os; return os.path.expanduser('~')

    def cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear

        yaw = quat_to_yaw(q.w, q.x, q.y, q.z)
        speed_xy = norm2(v.x, v.y)

        self.get_logger().info(
            f"x: {p.x}, y: {p.y}, yaw: {yaw}, vx: {v.x}, vy: {v.y}, vz: {v.z}, speed_xy: {speed_xy}"
        )
        
def main():
    rclpy.init(); n = GTPoseTap()
    try: rclpy.spin(n)
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
