#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from gt_tap.gt_utils import csv_logger, ensure_dir, quat_to_yaw, norm2

class GTPoseTap(Node):
    def __init__(self):
        super().__init__('gt_pose_tap')
        topic = '/ground_truth/odom'
        self.sub = self.create_subscription(Odometry, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing: {topic}")

        log_dir = ensure_dir(f'{self._home()}/eufs_dev/gt_data/logs')
        self.write = csv_logger(f'{log_dir}/gt_pose.csv',
                                ['stamp','frame','x','y','z','yaw','vx','vy','vz','speed_xy'])
        # (Optional) publish scalar speed for quick plotting
        self.pub_speed = self.create_publisher(Float32, '/gt/speed_xy', 10)

    def _home(self):
        import os; return os.path.expanduser('~')

    def cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear

        yaw = quat_to_yaw(q.w, q.x, q.y, q.z)
        speed_xy = norm2(v.x, v.y)

        self.write({
            'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9,
            'frame': msg.header.frame_id,
            'x': p.x, 'y': p.y, 'z': p.z,
            'yaw': yaw,
            'vx': v.x, 'vy': v.y, 'vz': v.z,
            'speed_xy': speed_xy
        })

        m = Float32(); m.data = float(speed_xy)
        self.pub_speed.publish(m)

def main():
    rclpy.init(); n = GTPoseTap()
    try: rclpy.spin(n)
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
