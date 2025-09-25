#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from gt_tap.gt_utils import ensure_dir, csv_logger, rosmsg_to_native, json_str

def _import_cones_msg():
    try:
        from eufs_msgs.msg import ConeArrayWithCovariance as ConesMsg  # common in EUFS
        return ConesMsg
    except Exception:
        try:
            from eufs_msgs.msg import ConeArray as ConesMsg
            return ConesMsg
        except Exception:
            raise RuntimeError("No eufs_msgs cones message found. Source EUFS.")

class GTConesTap(Node):
    def __init__(self):
        super().__init__('gt_cones_tap')
        ConesMsg = _import_cones_msg()
        topic = '/ground_truth/cones'
        self.sub = self.create_subscription(ConesMsg, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing: {topic}")

        base = f'{self._home()}/eufs_dev/gt_data/logs'
        ensure_dir(base)
        self.write_json = csv_logger(f'{base}/gt_cones.jsonl.csv', ['stamp','payload_json'])
        self.write_flat = csv_logger(f'{base}/gt_cones_flat.csv', ['stamp','x','y','z','color'])

    def _home(self): import os; return os.path.expanduser('~')

    def cb(self, msg):
        # JSONL-ish
        payload = rosmsg_to_native(msg)
        h = payload.get('header', {})
        ts = h.get('stamp', {}).get('sec', 0) + h.get('stamp', {}).get('nanosec', 0)*1e-9
        self.write_json({'stamp': ts, 'payload_json': json_str(payload)})

        # Flat CSV best-effort (works for ConeArray or ConeArrayWithCovariance)
        # We look for fields commonly named 'blue_cones', 'yellow_cones', 'orange_cones', 'big_orange_cones'
        for color_key in ['blue_cones','yellow_cones','orange_cones','big_orange_cones']:
            cones = payload.get(color_key, [])
            for c in cones:
                p = c.get('point', c)  # some schemas: {point:{x,y,z}, covariance:[]}
                x,y,z = p.get('x',0.0), p.get('y',0.0), p.get('z',0.0)
                self.write_flat({'stamp': ts, 'x': x, 'y': y, 'z': z, 'color': color_key})

def main():
    rclpy.init(); n = GTConesTap()
    try: rclpy.spin(n)
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
