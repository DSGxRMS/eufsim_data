#!/usr/bin/env python3
"""
odom_path_recorder.py

ROS2 node that:
- Subscribes to an Odometry topic (default: /ground_truth/odom)
- Records (x, y) positions whenever the point is at least `min_dist_m`
  away (Euclidean) from ALL previously stored points
- On shutdown (Ctrl+C), writes the recorded points to a CSV file in the
  current working directory, preserving order.
"""

import os
import math
import csv

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Odometry


def is_finite(*vals):
    return all(math.isfinite(v) for v in vals)


class OdomPathRecorder(Node):
    def __init__(self):
        super().__init__("odom_path_recorder", automatically_declare_parameters_from_overrides=True)

        # --- Parameters ---
        # Topic to subscribe to for odometry
        self.declare_parameter("topics.odom_in", "/ground_truth/odom")
        # Minimum Euclidean distance between any two stored points (meters)
        self.declare_parameter("min_dist_m", 1.0)
        # CSV file name to write in current working directory on shutdown
        self.declare_parameter("csv_filename", "recorded_path.csv")

        p = self.get_parameter
        self.odom_topic_in = str(p("topics.odom_in").value)
        self.min_dist_m = float(p("min_dist_m").value)
        self.min_dist_sq = self.min_dist_m ** 2
        self.csv_filename = str(p("csv_filename").value)

        # Buffer for recorded points: list of (x, y)
        self.points = []

        # QoS: reliable, small depth – ground-truth odom should be reliable
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Subscription
        self.create_subscription(Odometry, self.odom_topic_in, self.cb_odom, qos)

        self.get_logger().info(
            f"[odom_path_recorder] Subscribing to {self.odom_topic_in}, "
            f"min_dist_m={self.min_dist_m:.2f}, csv_filename='{self.csv_filename}'"
        )

    # ---------- Odometry callback ----------
    def cb_odom(self, msg: Odometry):
        # Extract x, y from pose
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        if not is_finite(x, y):
            return

        if self._should_add_point(x, y):
            self.points.append((x, y))
            self.get_logger().debug(
                f"Recorded point #{len(self.points)}: x={x:.3f}, y={y:.3f}"
            )

    def _should_add_point(self, x: float, y: float) -> bool:
        """Return True if (x, y) is at least min_dist_m away from ALL stored points."""
        if not self.points:
            # Always accept the first point
            return True

        for (px, py) in self.points:
            dx = x - px
            dy = y - py
            if (dx * dx + dy * dy) < self.min_dist_sq:
                # Too close to an existing point, reject
                return False

        return True

    # ---------- CSV writing on shutdown ----------
    def write_csv(self):
        if not self.points:
            self.get_logger().warn("[odom_path_recorder] No points recorded; nothing to write.")
            return

        path = os.path.join(os.getcwd(), self.csv_filename)
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                for (x, y) in self.points:
                    writer.writerow([x, y])
        except Exception as e:
            self.get_logger().error(f"Failed to write CSV to '{path}': {e}")
            return

        self.get_logger().info(
            f"[odom_path_recorder] Wrote {len(self.points)} points to CSV: {path}"
        )


def main():
    rclpy.init()
    node = OdomPathRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # On Ctrl+C or shutdown: write buffered points to CSV
        node.write_csv()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
