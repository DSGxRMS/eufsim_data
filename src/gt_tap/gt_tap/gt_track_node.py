#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy, json
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from gt_tap.gt_utils import ensure_dir, csv_logger, rosmsg_to_native, json_str

def _import_track_msg():
    try:
        from eufs_msgs.msg import TrackArray as TrackMsg
        return TrackMsg
    except Exception:
        raise RuntimeError("eufs_msgs.msg.TrackArray not found. Is EUFS sourced?")

class GTTrackTap(Node):
    def __init__(self):
        super().__init__('gt_track_tap')
        TrackMsg = _import_track_msg()
        topic = '/ground_truth/track'
        self.sub = self.create_subscription(TrackMsg, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing: {topic}")

        log_dir = ensure_dir(f'{self._home()}/eufs_dev/gt_data/logs')
        self.write = csv_logger(f'{log_dir}/gt_track.csv', ['stamp','payload_json'])

    def _home(self): import os; return os.path.expanduser('~')

    def cb(self, msg):
        payload = rosmsg_to_native(msg)
        h = payload.get('header', {})
        ts = h.get('stamp', {}).get('sec', 0) + h.get('stamp', {}).get('nanosec', 0)*1e-9
        self.write({'stamp': ts, 'payload_json': json_str(payload)})

def main():
    rclpy.init(); n = GTTrackTap()
    try: rclpy.spin(n)
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
