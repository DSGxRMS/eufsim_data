#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy, json, os
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Header
from gt_tap.gt_utils import ensure_dir, csv_logger, rosmsg_to_native, json_str

# We don't hardcode EUFS message type; import at runtime:
def _import_state_msg():
    # Commonly: eufs_msgs.msg.CarState (or similar)
    try:
        from eufs_msgs.msg import CarState as StateMsg
        return StateMsg
    except Exception:
        # fallback: try std_msgs/String (unlikely), else raise
        raise RuntimeError("Could not import eufs_msgs.msg.CarState. Source your EUFS workspace.")

class GTStateTap(Node):
    def __init__(self):
        super().__init__('gt_state_tap')
        StateMsg = _import_state_msg()
        topic = '/ground_truth/state'
        self.sub = self.create_subscription(StateMsg, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing: {topic}")

        log_dir = ensure_dir(f'{self._home()}/eufs_dev/gt_data/logs')
        self.write = csv_logger(f'{log_dir}/gt_state.csv', ['stamp','payload_json'])

    def _home(self): import os; return os.path.expanduser('~')

    def cb(self, msg):
        payload = rosmsg_to_native(msg)
        stamp = payload.get('header', {}).get('stamp', None)
        if isinstance(stamp, dict):
            ts = stamp.get('sec', 0) + stamp.get('nanosec', 0)*1e-9
        else:
            ts = self.get_clock().now().nanoseconds * 1e-9
        self.write({'stamp': ts, 'payload_json': json_str(payload)})

def main():
    rclpy.init(); n = GTStateTap()
    try: rclpy.spin(n)
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
