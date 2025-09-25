#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from eufs_msgs.msg import WheelSpeedsStamped 

class GTWheelsTap(Node):
    def __init__(self):
        super().__init__('gt_wheels_tap')
        topic = '/ground_truth/wheel_speeds'
        self.sub = self.create_subscription(WheelSpeedsStamped, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing: {topic}")

    def cb(self, msg: WheelSpeedsStamped):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        s  = msg.speeds
       
        self.get_logger().info(
            f"stamp:{ts:.6f}  steering:{s.steering:.3f}  "
            f"lf:{s.lf_speed:.3f}  rf:{s.rf_speed:.3f}  lb:{s.lb_speed:.3f}  rb:{s.rb_speed:.3f}"
        )

def main():
    rclpy.init()
    n = GTWheelsTap()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
