#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GT Cones visualizer (matplotlib)
- Subscribes to /ground_truth/cones (EUFS) with BEST_EFFORT QoS
- Plots BEV using matplotlib, color-coded by cone class
- Colors: yellow, blue, orange, big_orange (unknown -> gray)

Run:
  ros2 run <your_pkg> gt_cones_matplot.py
"""
import math
from typing import Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or leave default backend if already set
import matplotlib.pyplot as plt


# --------- Message import helper ---------
def _import_cones_msg():
    """Prefer ConeArrayWithCovariance; fall back to ConeArray."""
    try:
        from eufs_msgs.msg import ConeArrayWithCovariance as ConesMsg
        return ConesMsg
    except Exception:
        from eufs_msgs.msg import ConeArray as ConesMsg
        return ConesMsg


def _xyz_from_cone(cone) -> Tuple[float, float, float]:
    """
    Try to extract (x,y,z) robustly from various EUFS cone message representations.
    """
    # Common paths seen in EUFS stacks
    for attr in ("point", "position", "location"):
        if hasattr(cone, attr):
            p = getattr(cone, attr)
            return float(getattr(p, "x", 0.0)), float(getattr(p, "y", 0.0)), float(getattr(p, "z", 0.0))
    # Sometimes cones may directly have x,y,z
    if hasattr(cone, "x") and hasattr(cone, "y"):
        return float(cone.x), float(cone.y), float(getattr(cone, "z", 0.0))
    # Fallback
    return 0.0, 0.0, 0.0


class GTConesTap(Node):
    def __init__(self):
        super().__init__('gt_cones_tap')

        # ---- Topic & QoS (best effort) ----
        ConesMsg = _import_cones_msg()
        topic = '/ground_truth/cones'

        # qos_profile_sensor_data already is depth=5, best-effort, volatile, keep-last.
        # If you prefer explicit:
        # qos = QoSProfile(
        #     reliability=QoSReliabilityPolicy.BEST_EFFORT,
        #     history=QoSHistoryPolicy.KEEP_LAST,
        #     depth=10,
        #     durability=QoSDurabilityPolicy.VOLATILE
        # )
        qos = qos_profile_sensor_data

        self.sub = self.create_subscription(ConesMsg, topic, self.cb, qos)
        self.get_logger().info(f"Subscribing (BEST_EFFORT): {topic}")

        # ---- Viewer params (ROS parameters) ----
        self.declare_parameter('viz', True)                  # enable/disable plot window
        self.declare_parameter('range_x', [-10.0, 40.0])     # meters forward/back (x)
        self.declare_parameter('range_y', [-15.0, 15.0])     # meters left/right (y)
        self.declare_parameter('hz', 20.0)                   # plot refresh rate

        self.viz_enabled = bool(self.get_parameter('viz').value)
        rx = self.get_parameter('range_x').value
        ry = self.get_parameter('range_y').value
        self.range_x = (float(rx[0]), float(rx[1]))
        self.range_y = (float(ry[0]), float(ry[1]))
        self.hz = float(self.get_parameter('hz').value)

        # Cache latest cones by class for drawing
        self._last_cones = {
            'yellow_cones': [],
            'blue_cones': [],
            'orange_cones': [],
            'big_orange_cones': [],
            'unknown_color_cones': [],
        }

        # ---- Matplotlib setup ----
        self._fig = None
        self._ax = None
        self._scatters = {}  # class -> PathCollection

        if self.viz_enabled:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(7.5, 6))
            self._ax.set_title("GT Cones — BEV (x forward, y left)")
            self._ax.set_xlabel("x (m, forward)")
            self._ax.set_ylabel("y (m, left)")
            self._ax.set_aspect('equal', adjustable='box')
            self._ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)
            self._ax.set_xlim(self.range_x[0], self.range_x[1])
            self._ax.set_ylim(self.range_y[0], self.range_y[1])

            # Colors (matplotlib named colors)
            self._colors = {
                'blue_cones': 'tab:blue',
                'yellow_cones': 'gold',
                'orange_cones': 'orange',
                'big_orange_cones': 'darkorange',
                'unknown_color_cones': '0.6',   # gray
            }

            # Create one scatter per class
            for k, c in self._colors.items():
                self._scatters[k] = self._ax.scatter([], [], s=20, c=c, label=k.replace('_', ' '))

            # Car at origin marker
            self._ax.scatter([0.0], [0.0], s=60, facecolors='none', edgecolors='lime', linewidths=1.5, label='car (0,0)')

            self._ax.legend(loc='upper right', fontsize=8, frameon=True)

            # Timer to refresh plot
            self.timer = self.create_timer(1.0 / max(self.hz, 1.0), self._on_timer)
            self.get_logger().info("Matplotlib viewer started. Close the window to stop visualization.")

    # --- ROS callback ---
    def cb(self, msg):
        # Unpack by color classes present in EUFS message
        color_fields = [
            'yellow_cones',
            'blue_cones',
            'orange_cones',
            'big_orange_cones',
            'unknown_color_cones'
        ]

        counts = {}
        total = 0
        updated = {k: [] for k in color_fields}

        for key in color_fields:
            cones = getattr(msg, key, [])
            counts[key] = len(cones)
            total += counts[key]
            for c in cones:
                updated[key].append(_xyz_from_cone(c))

        self._last_cones = updated

        # Log a compact line
        try:
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception:
            ts = float('nan')
        parts = [f"t={ts:.3f}s", f"total={total}"] + [f"{k}={counts[k]}" for k in color_fields]
        self.get_logger().info(" | ".join(parts))

    # --- Plot refresh ---
    def _on_timer(self):
        if not self.viz_enabled:
            return
        if self._fig is None or self._ax is None:
            return
        # If figure closed by user, stop updating
        if not plt.fignum_exists(self._fig.number):
            self.get_logger().info("Viewer closed.")
            self.viz_enabled = False
            return

        # Update per-class scatter offsets
        for key, scatter in self._scatters.items():
            pts = self._last_cones.get(key, [])
            if len(pts) == 0:
                scatter.set_offsets(np.empty((0, 2)))
            else:
                arr = np.asarray([(x, y) for (x, y, _z) in pts], dtype=float)
                scatter.set_offsets(arr)

        # Light ticks every 5 m
        self._ax.set_xlim(self.range_x[0], self.range_x[1])
        self._ax.set_ylim(self.range_y[0], self.range_y[1])

        self._fig.canvas.draw_idle()
        plt.pause(0.001)  # yields to UI loop

    def destroy_node(self):
        # Close figure if still open
        try:
            if self._fig is not None and plt.fignum_exists(self._fig.number):
                plt.close(self._fig)
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = GTConesTap()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
