#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix
from gt_tap.gt_utils import ensure_dir, csv_logger

class GTGPSTap(Node):
    def __init__(self):
        super().__init__('gt_gps_tap')
        topic = '/gps'
        self.sub = self.create_subscription(NavSatFix, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing: {topic}")

        base = f'{self._home()}/eufs_dev/gt_data/logs'
        self.write = csv_logger(f'{base}/gt_gps.csv',
                                ['stamp','lat','lon','alt','status'])

    def _home(self): import os; return os.path.expanduser('~')

    def cb(self, msg: NavSatFix):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        status = getattr(msg.status, 'status', 0)
        self.write({'stamp': ts, 'lat': msg.latitude, 'lon': msg.longitude, 'alt': msg.altitude, 'status': status})

def main():
    rclpy.init(); n = GTGPSTap()
    try: rclpy.spin(n)
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
