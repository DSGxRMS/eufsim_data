#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiveORB + GT cones BEV visualizer (Qt5Agg backend)

- Runs stereo+YOLO+ORB to triangulate cones in camera -> vehicle -> world frame
- Subscribes to /ground_truth/cones (EUFS) and transforms GT cones to world frame using odom
- Plots BEV with Matplotlib:
    - Live cones: solid markers in class colors
    - GT cones: hollow markers with thick outlines in class colors
- Prints live cone array periodically to terminal.

Run:
  ros2 run <your_pkg> liveorb_gt_bev.py
"""

import threading
from typing import Tuple, List

import math
import time
from pathlib import Path

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import Odometry

import matplotlib  # backend set later, AFTER YOLO loads

# ----------------- Parameters -----------------
LEFT_IMAGE_TOPIC = "/zed/left/image_rect_color"
RIGHT_IMAGE_TOPIC = "/zed/right/image_rect_color"
ODOM_TOPIC = "/ground_truth/odom"
GT_CONES_TOPIC = "/ground_truth/cones"

YOLO_REPO_REL = "yolov5"
YOLO_WEIGHTS_REL = "yolov5/weights/best.pt"

# Camera intrinsics
FX = 448.13386274345095
FY = 448.13386274345095
CX = 640.5
CY = 360.5
BASELINE = 0.06  # meters

# ORB / matching params
ORB_N_FEATURES = 2500
LOWE_RATIO = 0.8
MIN_MATCHES_PER_BOX = 3

# Console output frequency (Hz)
OUTPUT_HZ = 5.0

# Plot refresh frequency (Hz)
PLOT_HZ = 90.0

# BEV plotting bounds (vehicle-centered coordinates)
BEV_XLIM = (-5.0, 40.0)   # forward
BEV_YLIM = (-15.0, 15.0)  # left/right

# Class ids -> names
CLASS_NAMES = {
    0: "Blue",
    1: "Yellow",
    2: "Orange",
    3: "Big Orange",
    4: "Unknown",
    -1: "Proximity Group",
}

# Class ids -> matplotlib colors
CLASS_COLORS_MPL = {
    0: "tab:blue",
    1: "gold",
    2: "orange",
    3: "darkorange",
    4: "0.6",
    -1: "tab:red",
}

# ----------------- Helpers for GT cones -----------------


def import_cones_msg():
    """
    Robust import for EUFS cone messages.
    Prefer ConeArrayWithCovariance; fall back to ConeArray.
    """
    try:
        from eufs_msgs.msg import ConeArrayWithCovariance as ConesMsg
        return ConesMsg
    except Exception:
        try:
            from eufs_msgs.msg import ConeArray as ConesMsg
            return ConesMsg
        except Exception:
            return None


def xyz_from_cone(cone) -> Tuple[float, float, float]:
    """
    Try to extract (x,y,z) robustly from various EUFS cone message representations.
    """
    for attr in ("point", "position", "location"):
        if hasattr(cone, attr):
            p = getattr(cone, attr)
            return float(getattr(p, "x", 0.0)), float(getattr(p, "y", 0.0)), float(getattr(p, "z", 0.0))
    if hasattr(cone, "x") and hasattr(cone, "y"):
        return float(cone.x), float(cone.y), float(getattr(cone, "z", 0.0))
    return 0.0, 0.0, 0.0


# ----------------- YOLO loader -----------------
class YOLOv5Detector:
    """
    YOLOv5 loader compatible with a local repo:
      - First try torch.hub.load(local_repo, 'custom', path=weights, source='local')
      - Fallback: in-repo DetectMultiBackend

    Public API:
        infer(img_rgb: np.ndarray) -> List[dict(x1,y1,x2,y2,conf,cls)]
    """

    def __init__(self, repo_dir: Path, weights_path: Path, device: str = "cpu"):
        import torch

        self.repo_dir = Path(repo_dir).resolve()
        self.weights_path = Path(weights_path).resolve()
        self.device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self.backend = None
        self.input_size = 640
        self.conf_thres = 0.35
        self.iou_thres = 0.45
        self.names = {}

        # Try torch.hub local load (yolov5)
        try:
            self.model = torch.hub.load(
                str(self.repo_dir),
                "custom",
                path=str(self.weights_path),
                source="local",
            ).to(self.device)
            self.model.eval()
            self.names = self.model.names if hasattr(self.model, "names") else {}
            self.backend = "hub"
            print("[YOLO] Loaded via torch.hub (local).")
            return
        except Exception as e:
            print("[YOLO] torch.hub load failed:", e)

        # Try raw in-repo loading (DetectMultiBackend)
        try:
            import sys
            sys.path.insert(0, str(self.repo_dir))
            from models.common import DetectMultiBackend
            from utils.augmentations import letterbox
            from utils.general import non_max_suppression, scale_coords

            self.DetectMultiBackend = DetectMultiBackend
            self.letterbox = letterbox
            self.nms = non_max_suppression
            self.scale_coords = scale_coords
            self.model = DetectMultiBackend(str(self.weights_path), device=self.device)
            self.names = getattr(self.model, "names", {})
            self.backend = "raw"
            print("[YOLO] Loaded via raw DetectMultiBackend.")
            return
        except Exception as e:
            print("[YOLO] raw backend load failed:", e)
            raise RuntimeError("Failed to load YOLOv5 model. Check repo & weights paths.")

    def infer(self, img_rgb: np.ndarray):
        """
        Return list of detections: dict(x1,y1,x2,y2,conf,cls)
        """
        if img_rgb is None:
            return []

        if self.backend == "hub":
            import torch
            results = self.model(img_rgb, size=self.input_size)
            dets = []
            if hasattr(results, "xyxy") and len(results.xyxy):
                preds = results.xyxy[0].cpu().numpy()
                for x1, y1, x2, y2, conf, cls in preds:
                    if float(conf) < self.conf_thres:
                        continue
                    dets.append(
                        {
                            "x1": int(round(x1)),
                            "y1": int(round(y1)),
                            "x2": int(round(x2)),
                            "y2": int(round(y2)),
                            "conf": float(conf),
                            "cls": int(cls),
                        }
                    )
            return dets

        elif self.backend == "raw":
            import torch
            im = self.letterbox(img_rgb, new_shape=self.input_size)[0]
            im = im.transpose(2, 0, 1)
            im = np.ascontiguousarray(im)

            im_t = torch.from_numpy(im).to(self.device).float() / 255.0
            if im_t.ndim == 3:
                im_t = im_t.unsqueeze(0)

            pred = self.model(im_t)
            pred = self.nms(pred, self.conf_thres, self.iou_thres)[0]
            out = []
            if pred is None:
                return out

            pred[:, :4] = self.scale_coords(im_t.shape[2:], pred[:, :4], img_rgb.shape).round()
            for *xyxy, conf, cls in pred.cpu().tolist():
                x1, y1, x2, y2 = map(int, xyxy)
                out.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "conf": float(conf),
                        "cls": int(cls),
                    }
                )
            return out

        else:
            return []


# ----------------- Main Node -----------------
class LiveORBGTBEVNode(Node):
    def __init__(self):
        super().__init__('liveorb_gt_bev')

        self.get_logger().info("Starting LiveORB + GT BEV visualizer node")

        # ---- Parameters ----
        self.declare_parameter('left_image_topic', LEFT_IMAGE_TOPIC)
        self.declare_parameter('right_image_topic', RIGHT_IMAGE_TOPIC)
        self.declare_parameter('odom_topic', ODOM_TOPIC)
        self.declare_parameter('gt_cones_topic', GT_CONES_TOPIC)
        self.declare_parameter('yolo_repo', YOLO_REPO_REL)
        self.declare_parameter('yolo_weights', YOLO_WEIGHTS_REL)
        self.declare_parameter('baseline', BASELINE)
        self.declare_parameter('fx', FX)
        self.declare_parameter('fy', FY)
        self.declare_parameter('cx', CX)
        self.declare_parameter('cy', CY)
        self.declare_parameter('viz', True)
        self.declare_parameter('plot_hz', PLOT_HZ)
        self.declare_parameter('range_x', list(BEV_XLIM))
        self.declare_parameter('range_y', list(BEV_YLIM))

        self.left_topic = self.get_parameter('left_image_topic').value
        self.right_topic = self.get_parameter('right_image_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.gt_topic = self.get_parameter('gt_cones_topic').value

        base = Path(__file__).parent.resolve()
        repo_dir = base / self.get_parameter('yolo_repo').value
        weights_path = base / self.get_parameter('yolo_weights').value

        self.fx = float(self.get_parameter('fx').value)
        self.fy = float(self.get_parameter('fy').value)
        self.cx = float(self.get_parameter('cx').value)
        self.cy = float(self.get_parameter('cy').value)
        self.baseline = float(self.get_parameter('baseline').value)

        self.viz_enabled = bool(self.get_parameter('viz').value)
        rx = self.get_parameter('range_x').value
        ry = self.get_parameter('range_y').value
        self.range_x = (float(rx[0]), float(rx[1]))
        self.range_y = (float(ry[0]), float(ry[1]))
        self.hz = float(self.get_parameter('plot_hz').value)

        # YOLO, ORB, BF, Bridge
        try:
            self.yolo = YOLOv5Detector(repo_dir, weights_path, device="cuda")
        except Exception as e:
            self.get_logger().warning(f"YOLO failed to load: {e}")
            self.yolo = None

        self.orb = cv2.ORB_create(ORB_N_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.bridge = CvBridge()

        # ---------- Matplotlib backend: force Qt5Agg AFTER YOLO imports ----------
        matplotlib.use("Qt5Agg", force=True)
        import matplotlib.pyplot as plt
        self.plt = plt
        # ------------------------------------------------------------------------

        # --- Shared state ---
        self.data_lock = threading.Lock()
        self.left_img = None
        self.right_img = None
        self.odom_pose = (0.0, 0.0, 0.0)
        self.live_cones_world: List[Tuple[float, float, float, int]] = []
        self.gt_cones_world: List[Tuple[float, float, int]] = []

        # ---- Subscribers ----
        qos = qos_profile_sensor_data
        self.left_sub = self.create_subscription(RosImage, self.left_topic, self.left_cb, qos)
        self.right_sub = self.create_subscription(RosImage, self.right_topic, self.right_cb, qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos)

        ConesMsg = import_cones_msg()
        if ConesMsg is not None:
            self.gt_sub = self.create_subscription(ConesMsg, self.gt_topic, self.gt_cb, qos)
            self.get_logger().info(f"Subscribed to GT cones: {self.gt_topic}")
        else:
            self.gt_sub = None
            self.get_logger().warning("eufs_msgs ConeArray not available; GT overlay disabled")

        # ---- Output timer (printing) ----
        self.output_timer = self.create_timer(1.0 / OUTPUT_HZ, self._on_output_timer)

        # ---- Matplotlib setup (GTConesTap-style) ----
        self._fig = None
        self._ax = None
        self._live_scatters = {}  # cls_id -> PathCollection (solid)
        self._gt_scatters = {}    # cls_id -> PathCollection (hollow)

        if self.viz_enabled:
            self.plt.ion()
            self._fig, self._ax = self.plt.subplots(figsize=(7.5, 6))
            try:
                self._fig.canvas.manager.set_window_title("LiveORB + GT Cones — BEV")
            except Exception:
                pass

            self._ax.set_title("LiveORB + GT Cones — BEV (x forward, y left)")
            self._ax.set_xlabel("x (m, forward)")
            self._ax.set_ylabel("y (m, left)")
            self._ax.set_aspect('equal', adjustable='box')
            self._ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)
            self._ax.set_xlim(self.range_x[0], self.range_x[1])
            self._ax.set_ylim(self.range_y[0], self.range_y[1])

            # Create per-class scatters
            for cls_id, color in CLASS_COLORS_MPL.items():
                self._live_scatters[cls_id] = self._ax.scatter(
                    [], [], s=30, c=color, marker='o',
                    label=f"Live {CLASS_NAMES.get(cls_id, str(cls_id))}"
                )
                self._gt_scatters[cls_id] = self._ax.scatter(
                    [], [], s=60, facecolors='none', edgecolors=color,
                    linewidths=1.5, marker='o',
                    label=f"GT {CLASS_NAMES.get(cls_id, str(cls_id))}"
                )

            # Car at origin
            self._ax.scatter(
                [0.0], [0.0], s=60,
                facecolors='none', edgecolors='lime',
                linewidths=1.5, marker='s', label='car (0,0)'
            )

            # Legend
            self._ax.legend(loc='upper right', fontsize=8, frameon=True)

            # Plot timer
            self._plot_timer = self.create_timer(1.0 / max(self.hz, 1.0), self._on_plot_timer)
            self.get_logger().info("Matplotlib BEV viewer started (Qt5Agg).")

        backend = self.plt.get_backend()
        self.get_logger().info(f"Matplotlib backend in node: {backend}")

        self.get_logger().info("Node initialized.")

    # ----------------- ROS Callbacks -----------------
    def left_cb(self, msg: RosImage):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.data_lock:
                self.left_img = cv_img.copy()
        except Exception as e:
            self.get_logger().error(f"left_cb cv_bridge error: {e}")

    def right_cb(self, msg: RosImage):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.data_lock:
                self.right_img = cv_img.copy()
        except Exception as e:
            self.get_logger().error(f"right_cb cv_bridge error: {e}")

    def odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        with self.data_lock:
            self.odom_pose = (float(px), float(py), float(yaw))

    def gt_cb(self, msg):
        color_fields = [
            ('yellow_cones', 1),
            ('blue_cones', 0),
            ('orange_cones', 2),
            ('big_orange_cones', 3),
            ('unknown_color_cones', 4),
        ]

        with self.data_lock:
            ox, oy, oyaw = self.odom_pose

        parsed = []
        for field, cls_id in color_fields:
            cones = getattr(msg, field, [])
            for c in cones:
                x, y, z = xyz_from_cone(c)  # car frame: x forward, y left
                Xw = ox + x * math.cos(oyaw) - y * math.sin(oyaw)
                Yw = oy + x * math.sin(oyaw) + y * math.cos(oyaw)
                parsed.append((float(Xw), float(Yw), int(cls_id)))

        with self.data_lock:
            self.gt_cones_world = parsed

    # ----------------- Core computation -----------------
    def _calculate_live_cones(
        self,
        imgL,
        imgR,
        odom_pose,
    ) -> List[Tuple[float, float, float, int]]:
        if imgL is None or imgR is None:
            return []

        ox, oy, oyaw = odom_pose

        # YOLO detections
        boxesL, boxesR = [], []
        if self.yolo is not None:
            try:
                detsL = self.yolo.infer(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
                detsR = self.yolo.infer(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
                boxesL = [(d['cls'], d['x1'], d['y1'], d['x2'], d['y2']) for d in detsL]
                boxesR = [(d['cls'], d['x1'], d['y1'], d['x2'], d['y2']) for d in detsR]
            except Exception as e:
                self.get_logger().warning(f"YOLO inference error: {e}")

        # ORB
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        kpL, desL = self.orb.detectAndCompute(grayL, None)
        kpR, desR = self.orb.detectAndCompute(grayR, None)
        if desL is None or desR is None:
            return []

        matches_knn = self.bf.knnMatch(desL, desR, k=2)

        good_matches_data = []
        for pair in matches_knn:
            if len(pair) > 1 and pair[0].distance < LOWE_RATIO * pair[1].distance:
                m = pair[0]
                xl, yl = kpL[m.queryIdx].pt
                xr, yr = kpR[m.trainIdx].pt
                good_matches_data.append((m, (xl, yl), (xr, yr)))

        matches_by_pair = {}

        for _m_obj, (xl, yl), (xr, yr) in good_matches_data:
            xl_i, yl_i, xr_i, yr_i = int(xl), int(yl), int(xr), int(yr)

            if boxesL and boxesR:
                for clsL, x1L, y1L, x2L, y2L in boxesL:
                    if x1L <= xl_i <= x2L and y1L <= yl_i <= y2L:
                        for clsR, x1R, y1R, x2R, y2R in boxesR:
                            if (
                                clsL == clsR
                                and x1R <= xr_i <= x2R
                                and y1R <= yr_i <= y2R
                            ):
                                key = (
                                    (x1L, y1L, x2L, y2L),
                                    (x1R, y1R, x2R, y2R),
                                    clsL,
                                )
                                matches_by_pair.setdefault(key, []).append(
                                    (
                                        (int(round(xl)), int(round(yl))),
                                        (int(round(xr)), int(round(yr))),
                                    )
                                )
                                break
                        break
            else:
                key = (
                    (int(xl) - 5, int(yl) - 5, int(xl) + 5, int(yl) + 5),
                    (int(xr) - 5, int(yr) - 5, int(xr) + 5, int(yr) + 5),
                    -1,
                )
                matches_by_pair.setdefault(key, []).append(
                    (
                        (int(round(xl)), int(round(yl))),
                        (int(round(xr)), int(round(yr))),
                    )
                )

        final_cones = []
        for (_, _, cls), pts in matches_by_pair.items():
            if len(pts) < MIN_MATCHES_PER_BOX:
                continue

            left_us = np.array([p[0][0] for p in pts], dtype=float)
            left_vs = np.array([p[0][1] for p in pts], dtype=float)
            right_us = np.array([p[1][0] for p in pts], dtype=float)

            uL = float(left_us.mean())
            vL = float(left_vs.mean())
            uR = float(right_us.mean())
            disparity = uL - uR
            if disparity <= 0.1:
                continue

            Z_cam = (self.fx * self.baseline) / disparity
            X_cam = (uL - self.cx) * Z_cam / self.fx

            Xv = float(Z_cam)       # forward
            Yv = float(-X_cam)      # left

            Xw = ox + Xv * math.cos(oyaw) - Yv * math.sin(oyaw)
            Yw = oy + Xv * math.sin(oyaw) + Yv * math.cos(oyaw)

            final_cones.append((float(Xw), float(Yw), float(Z_cam), int(cls)))

        return final_cones

    # ----------------- Plot timer (GTConesTap-style) -----------------
    def _on_plot_timer(self):
        if not self.viz_enabled:
            return
        if self._fig is None or self._ax is None:
            return
        if not self.plt.fignum_exists(self._fig.number):
            self.get_logger().info("Viewer closed.")
            self.viz_enabled = False
            return

        # Snapshot of data
        with self.data_lock:
            imgL_copy = self.left_img.copy() if self.left_img is not None else None
            imgR_copy = self.right_img.copy() if self.right_img is not None else None
            odom_pose_copy = self.odom_pose
            gt_copy = list(self.gt_cones_world)

        # Compute live cones
        live_cones = self._calculate_live_cones(imgL_copy, imgR_copy, odom_pose_copy)

        # Store for printing
        with self.data_lock:
            self.live_cones_world = live_cones

        ox, oy, oyaw = odom_pose_copy

        def world_to_vehicle(xw, yw):
            rx = xw - ox
            ry = yw - oy
            vx = rx * math.cos(-oyaw) - ry * math.sin(-oyaw)
            vy = rx * math.sin(-oyaw) + ry * math.cos(-oyaw)
            return vx, vy

        # Build per-class arrays
        live_dict = {cls_id: [] for cls_id in CLASS_COLORS_MPL.keys()}
        gt_dict = {cls_id: [] for cls_id in CLASS_COLORS_MPL.keys()}

        for (Xw, Yw, _Z_cam, cls_id) in live_cones:
            vx, vy = world_to_vehicle(Xw, Yw)
            live_dict.setdefault(cls_id, []).append((vx, vy))

        for (Xw, Yw, cls_id) in gt_copy:
            vx, vy = world_to_vehicle(Xw, Yw)
            gt_dict.setdefault(cls_id, []).append((vx, vy))

        # Update scatters similar to GTConesTap
        for cls_id in CLASS_COLORS_MPL.keys():
            # Live
            live_pts = live_dict.get(cls_id, [])
            if len(live_pts) == 0:
                self._live_scatters[cls_id].set_offsets(np.empty((0, 2)))
            else:
                arr = np.asarray(live_pts, dtype=float)
                self._live_scatters[cls_id].set_offsets(arr)

            # GT
            gt_pts = gt_dict.get(cls_id, [])
            if len(gt_pts) == 0:
                self._gt_scatters[cls_id].set_offsets(np.empty((0, 2)))
            else:
                arr = np.asarray(gt_pts, dtype=float)
                self._gt_scatters[cls_id].set_offsets(arr)

        self._ax.set_xlim(self.range_x[0], self.range_x[1])
        self._ax.set_ylim(self.range_y[0], self.range_y[1])

        self._fig.canvas.draw_idle()
        self.plt.pause(0.001)

    # ----------------- Output timer (printing) -----------------
    def _on_output_timer(self):
        with self.data_lock:
            cones_copy = list(self.live_cones_world)

        current_time = time.strftime("%H:%M:%S")

        print("\n" + "=" * 60)
        print(f"[{current_time}] Live cones ({len(cones_copy)})")
        print("-" * 60)

        if cones_copy:
            out = []
            for Xw, Yw, Z_cam, cls_id in cones_copy:
                cls_name = CLASS_NAMES.get(cls_id, "Unknown")
                out.append(f"[{Xw:.2f}, {Yw:.2f}, {cls_id} ({cls_name})]")
            print(f"[{', '.join(out)}]")
        else:
            print("No cones detected or waiting for data...")
        print("=" * 60)

    # ----------------- Cleanup -----------------
    def destroy_node(self):
        try:
            if hasattr(self, '_plot_timer'):
                self._plot_timer.cancel()
        except Exception:
            pass
        try:
            if hasattr(self, 'output_timer'):
                self.output_timer.cancel()
        except Exception:
            pass
        try:
            if self._fig is not None and self.plt.fignum_exists(self._fig.number):
                self.plt.close(self._fig)
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = LiveORBGTBEVNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
