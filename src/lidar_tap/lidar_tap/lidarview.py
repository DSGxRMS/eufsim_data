#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 viewer for /velodyne_points
Pipeline:
  - RANSAC plane (ground removal)
  - Euclidean clustering (KDTree)
  - Min/Max cluster-size filtering (1:1 with processing node)
  - 3D display (ground gray, clusters colored)

Run:
  ros2 run lidar_tap pcviewer
"""

import sys, time, math, random
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from sklearn.neighbors import KDTree

# GUI
from PySide6 import QtCore, QtWidgets
import pyqtgraph.opengl as gl


# ----------------- Euclidean clustering (KDTree BFS) -----------------
def euclidean_clustering(P: np.ndarray, radius: float):
    if P.size == 0:
        return []
    tree = KDTree(P)
    n_points = P.shape[0]
    visited = np.zeros(n_points, dtype=bool)
    clusters = []
    for idx in range(n_points):
        if visited[idx]:
            continue
        cluster = []
        queue = [idx]
        visited[idx] = True
        while queue:
            current_idx = queue.pop(0)
            cluster.append(current_idx)
            indices = tree.query_radius(P[current_idx].reshape(1, -1), r=radius)[0]
            for neighbor_idx in indices:
                ni = int(neighbor_idx)
                if not visited[ni]:
                    queue.append(ni)
                    visited[ni] = True
        clusters.append(cluster)
    return clusters


# ----------------- RANSAC plane (NumPy, same logic as processing node) -----------------
class RANSAC:
    def __init__(self, point_cloud: np.ndarray, max_iterations: int, distance_ratio_threshold: float):
        self.point_cloud = np.asarray(point_cloud, dtype=np.float32)
        self.max_iterations = int(max_iterations)
        self.distance_ratio_threshold = float(distance_ratio_threshold)

    def _ransac_algorithm(self):
        P = self.point_cloud
        n = P.shape[0]
        if n < 3:
            return P.copy(), np.empty((0, 3), dtype=P.dtype)

        inliers_best = set()
        iters = self.max_iterations

        while iters:
            iters -= 1
            # pick 3 unique indices
            inliers = []
            seen = set()
            while len(inliers) < 3:
                idx = random.randint(0, n - 1)
                if idx not in seen:
                    inliers.append(idx)
                    seen.add(idx)

            x1, y1, z1 = P[inliers[0]]
            x2, y2, z2 = P[inliers[1]]
            x3, y3, z3 = P[inliers[2]]

            a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
            b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
            c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            d = -(a * x1 + b * y1 + c * z1)
            denom = max(0.1, math.sqrt(a * a + b * b + c * c))

            # vectorized distance check
            dist = np.abs((P @ np.array([a, b, c], dtype=np.float32)) + d) / denom
            inside = np.where(dist <= self.distance_ratio_threshold)[0]
            cur = set(inliers).union(set(inside.tolist()))

            if len(cur) > len(inliers_best):
                inliers_best = cur

        mask = np.zeros(n, dtype=bool)
        if inliers_best:
            mask[list(inliers_best)] = True
        inlier_points = P[mask]
        outlier_points = P[~mask]
        return inlier_points, outlier_points


# ----------------- ROS + Viewer -----------------
class VelodyneViewer(Node):
    def __init__(self):
        super().__init__('pcviewer')
        topic = '/velodyne_points'
        self.sub = self.create_subscription(PointCloud2, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing to: {topic}")

        # pipeline knobs (1:1 with processing node)
        self.decimate = 1                # 1=off
        self.cluster_radius = 0.02        # meters
        self.min_cluster_sz = 10         # <-- same as node
        self.max_cluster_sz = 30# <-- same as node
        self.ransac_iters = 5
        self.ransac_thresh = 0.01        # meters

        # state
        self._pts = np.empty((0, 3), dtype=np.float32)
        self._recv = 0
        self._frames = 0
        self._last = time.time()

        # Qt window
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = QtWidgets.QMainWindow()
        self.win.resize(1280, 840)
        self.win.setWindowTitle("Velodyne Viewer — ground removed + clusters")
        central = QtWidgets.QWidget(self.win)
        self.win.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=35)
        layout.addWidget(self.view, 1)
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        grid.setSize(100, 100)
        self.view.addItem(grid)

        self.lbl = QtWidgets.QLabel("Msgs:0  FPS:--  Raw:--  Ground:--  NG:--  Clusters:--")
        layout.addWidget(self.lbl, 0)

        # draw layers
        self.pcd_ground = gl.GLScatterPlotItem(); self.pcd_ground.setGLOptions('opaque'); self.view.addItem(self.pcd_ground)
        self.pcd_ng     = gl.GLScatterPlotItem(); self.pcd_ng.setGLOptions('opaque');     self.view.addItem(self.pcd_ng)
        self.pcd_cent   = gl.GLScatterPlotItem(); self.pcd_cent.setGLOptions('opaque');   self.view.addItem(self.pcd_cent)

        # tick
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)

    # ROS callback
    def cb(self, msg: PointCloud2):
        self._recv += 1
        try:
            gen = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            pts = np.fromiter((c for xyz in gen for c in xyz), dtype=np.float32)
            if pts.size:
                pts = pts.reshape(-1, 3)
                if self.decimate > 1:
                    pts = pts[::int(self.decimate)]
                self._pts = pts
        except Exception as e:
            self.get_logger().warn(f"pc2 parse error: {e}")

    # periodic processing + draw
    def _tick(self):
        rclpy.spin_once(self, timeout_sec=0.0)

        P = self._pts
        raw_n = int(P.shape[0])

        if raw_n == 0:
            self._set(self.pcd_ground, np.empty((0, 3)))
            self._set(self.pcd_ng, np.empty((0, 3)))
            self._set(self.pcd_cent, np.empty((0, 3)))
            self._hud(raw_n, 0, 0, 0)
            return

        # ---- Ground removal (RANSAC on NumPy) ----
        algo = RANSAC(P, max_iterations=self.ransac_iters, distance_ratio_threshold=self.ransac_thresh)
        G, NG = algo._ransac_algorithm()
        if G.size == 0:
            G = np.empty((0, 3), dtype=np.float32)
        if NG.size == 0:
            NG = np.empty((0, 3), dtype=np.float32)

        # ---- Euclidean clustering (KDTree-based BFS) ----
        clusters = euclidean_clustering(NG, self.cluster_radius) if NG.size else []

        # ---- Apply min/max cluster-size filtering (1:1 logic) ----
        if self.min_cluster_sz is not None or self.max_cluster_sz is not None:
            clusters = [
                c for c in clusters
                if (self.min_cluster_sz is None or len(c) >= self.min_cluster_sz) and
                   (self.max_cluster_sz is None or len(c) <= self.max_cluster_sz)
            ]

        # ---- Colorize + centroids ----
        centroids = []
        if clusters:
            colors = np.tile(np.array([[0.2, 0.6, 1.0, 0.9]], dtype=np.float32), (NG.shape[0], 1))
            for ci, idxs in enumerate(clusters):
                idxs = np.asarray(idxs, dtype=int)
                c = ((ci * 37) % 255) / 255.0
                colors[idxs] = np.array([c, 1.0 - c, 0.3, 0.95], dtype=np.float32)
                centroids.append(NG[idxs].mean(axis=0))
            self.pcd_ng.setData(pos=NG, size=2.0, color=colors)
        else:
            self._set(self.pcd_ng, NG, size=2.0)

        C = np.vstack(centroids).astype(np.float32) if len(centroids) else np.empty((0, 3), dtype=np.float32)

        # ---- draw layers ----
        self._set(self.pcd_ground, G, size=1.2, color=(0.4, 0.4, 0.4, 0.6))
        self._set(self.pcd_cent, C, size=7.0, color=(1, 0, 0, 1))

        self._hud(raw_n, G.shape[0], NG.shape[0], len(clusters))

    def _hud(self, raw_n, g_n, ng_n, ncl):
        self._frames += 1
        now = time.time()
        if now - self._last >= 0.5:
            fps = self._frames / (now - self._last)
            self._frames = 0
            self._last = now
            self.lbl.setText(f"Msgs:{self._recv}  FPS:{fps:4.1f}  Raw:{raw_n:6d}  Ground:{g_n:6d}  NG:{ng_n:6d}  Clusters:{ncl:3d}")
            self.win.setWindowTitle(f"Velodyne Viewer — {fps:4.1f} FPS")

    @staticmethod
    def _set(item, arr, size=None, color=None):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros((1, 3), dtype=np.float32)
        kw = {}
        if size is not None:
            kw["size"] = float(size)
        if color is not None:
            kw["color"] = color
        item.setData(pos=arr, **kw)


def main():
    rclpy.init()
    node = VelodyneViewer()
    node.win.show()
    try:
        sys.exit(node.app.exec())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
