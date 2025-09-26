#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

# Optional viewer
import cv2
import numpy as np

def _import_cones_msg():
    # Prefer WithCovariance; fall back to ConeArray if needed
    try:
        from eufs_msgs.msg import ConeArrayWithCovariance as ConesMsg
        return ConesMsg
    except Exception:
        from eufs_msgs.msg import ConeArray as ConesMsg
        return ConesMsg  # <-- FIX: return the fallback type, not None

class GTConesTap(Node):
    def __init__(self):
        super().__init__('gt_cones_tap')
        ConesMsg = _import_cones_msg()
        topic = '/ground_truth/cones'
        self.sub = self.create_subscription(ConesMsg, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing: {topic}")

        # ---- Viewer params (ROS parameters) ----
        self.declare_parameter('viz', True)                  # enable/disable window
        self.declare_parameter('bev_width', 800)
        self.declare_parameter('bev_height', 600)
        # meters of world coords shown in BEV (x forward, y left)
        self.declare_parameter('range_x', [-10.0, 40.0])     # forward/back
        self.declare_parameter('range_y', [-15.0, 15.0])     # left/right

        self.viz_enabled = bool(self.get_parameter('viz').value)
        self.bev_w = int(self.get_parameter('bev_width').value)
        self.bev_h = int(self.get_parameter('bev_height').value)
        rx = self.get_parameter('range_x').value
        ry = self.get_parameter('range_y').value
        self.range_x = (float(rx[0]), float(rx[1]))
        self.range_y = (float(ry[0]), float(ry[1]))

        # Precompute scales for BEV mapping
        self._sx = self.bev_w / (self.range_x[1] - self.range_x[0])
        self._sy = self.bev_h / (self.range_y[1] - self.range_y[0])

        self._last_cones = None  # cache latest cones per class for drawing

        if self.viz_enabled:
            self.get_logger().info("Viewer window started. Press 'q' in the window to close.")
            self.timer = self.create_timer(0.05, self._on_timer)  # ~20 Hz

    # --- Utilities ---
    @staticmethod
    def _xyz_from_cone(cone):
        p = cone.point
        return float(p.x), float(p.y), float(p.z)

    def _map_bev(self, x, y):
        u = int((x - self.range_x[0]) * self._sx)
        v = int((y - self.range_y[0]) * self._sy)
        return u, self.bev_h - v  # flip vertically so +y (left) appears to the left

    def _draw_bev(self, cones_dict):
        img = np.zeros((self.bev_h, self.bev_w, 3), dtype=np.uint8)

        # grid every 5m
        for gx in np.arange(math.ceil(self.range_x[0]/5)*5, self.range_x[1]+1e-3, 5.0):
            u0, v0 = self._map_bev(gx, self.range_y[0])
            u1, v1 = self._map_bev(gx, self.range_y[1])
            cv2.line(img, (u0, v0), (u1, v1), (40, 40, 40), 1)
            cv2.putText(img, f"{gx:.0f}m", (u0+2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1, cv2.LINE_AA)

        for gy in np.arange(math.ceil(self.range_y[0]/5)*5, self.range_y[1]+1e-3, 5.0):
            u0, v0 = self._map_bev(self.range_x[0], gy)
            u1, v1 = self._map_bev(self.range_x[1], gy)
            cv2.line(img, (u0, v0), (u1, v1), (40, 40, 40), 1)
            cv2.putText(img, f"{gy:.0f}m", (2, v0-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1, cv2.LINE_AA)

        # color map (BGR)
        colors = {
            'blue_cones':         (255, 80,  60),
            'yellow_cones':       (40,  220, 255),
            'orange_cones':       (0,   140, 255),
            'big_orange_cones':   (0,   90,  255),
            'unknown_color_cones':(180, 180, 180),
        }

        # Draw cones
        radius = 3
        for key, pts in cones_dict.items():
            color = colors.get(key, (180, 180, 180))
            for (x,y,z) in pts:
                u,v = self._map_bev(x,y)
                if 0 <= u < self.bev_w and 0 <= v < self.bev_h:
                    cv2.circle(img, (u,v), radius, color, thickness=-1, lineType=cv2.LINE_AA)

        # Car at origin (0,0)
        u0, v0 = self._map_bev(0.0, 0.0)
        cv2.circle(img, (u0, v0), 6, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img, "car", (u0+8, v0-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        cv2.imshow("GT Cones — BEV", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Closing BEV viewer.")
            self.viz_enabled = False
            cv2.destroyAllWindows()
            if hasattr(self, 'timer'):
                self.timer.cancel()

    # --- ROS callback ---
    def cb(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        color_fields = ['blue_cones', 'yellow_cones', 'orange_cones', 'big_orange_cones', 'unknown_color_cones']

        counts = {}
        samples = {}
        cones_dict = {k: [] for k in color_fields}
        total = 0

        for key in color_fields:
            cones = getattr(msg, key, [])
            counts[key] = len(cones)
            total += counts[key]
            for c in cones:
                cones_dict[key].append(self._xyz_from_cone(c))
            samples[key] = cones_dict[key][:3]

        parts = [f"t={ts:.3f}s", f"total={total}"]
        parts += [f"{k}={counts[k]}" for k in color_fields]
        self.get_logger().info(" | ".join(parts))

        self._last_cones = cones_dict

    # --- Timer to draw at fixed rate ---
    def _on_timer(self):
        if not self.viz_enabled:
            return
        if self._last_cones is None:
            empty = {k: [] for k in ['blue_cones','yellow_cones','orange_cones','big_orange_cones','unknown_color_cones']}
            self._draw_bev(empty)
        else:
            self._draw_bev(self._last_cones)

def main():
    rclpy.init()
    n = GTConesTap()
    try:
        rclpy.spin(n)
    finally:
        if n.viz_enabled:
            cv2.destroyAllWindows()
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
