#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from gt_tap.gt_utils import ensure_dir, csv_logger, rosmsg_to_native

def _import_ws_msg():
    try:
        from eufs_msgs.msg import WheelSpeedsStamped as WSM
        return WSM
    except Exception:
        raise RuntimeError("eufs_msgs.msg.WheelSpeedsStamped not found.")

class GTWheelsTap(Node):
    def __init__(self):
        super().__init__('gt_wheels_tap')
        WSM = _import_ws_msg()
        topic = '/ground_truth/wheel_speeds'
        self.sub = self.create_subscription(WSM, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing: {topic}")

        base = f'{self._home()}/eufs_dev/gt_data/logs'
        self.write = csv_logger(f'{base}/gt_wheel_speeds.csv',
                                ['stamp','fl','fr','rl','rr'])

    def _home(self): import os; return os.path.expanduser('~')

    def cb(self, msg):
        # Common fields: msg.wheel_speeds.{fl,fr,rl,rr} or similar
        try:
            ws = msg.wheel_speeds
            fl = getattr(ws, 'fl', 0.0); fr = getattr(ws, 'fr', 0.0)
            rl = getattr(ws, 'rl', 0.0); rr = getattr(ws, 'rr', 0.0)
        except Exception:
            # fallback: try flat names
            fl = getattr(msg, 'fl', 0.0); fr = getattr(msg, 'fr', 0.0)
            rl = getattr(msg, 'rl', 0.0); rr = getattr(msg, 'rr', 0.0)

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        self.write({'stamp': ts, 'fl': fl, 'fr': fr, 'rl': rl, 'rr': rr})

def main():
    rclpy.init(); n = GTWheelsTap()
    try: rclpy.spin(n)
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
