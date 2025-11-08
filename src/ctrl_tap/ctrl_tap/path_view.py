#!/usr/bin/env python3
# path_window_plotter.py
#
# Live view:
#   - Car pose as a triangle (oriented by yaw)
#   - "Window" path points received from topic (dots)  [UNTOUCHED]
#   - ALL CSV track points overlaid as 'x' markers, but TRANSLATED so the FIRST CSV POINT is (0,0)
#
import math
from typing import Optional, List, Tuple
from pathlib import Path as FSPath

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PathWindowPlotter(Node):
    def __init__(self):
        super().__init__('path_window_plotter', automatically_declare_parameters_from_overrides=True)

        # -------- params --------
        self.declare_parameter('odom_topic', '/ground_truth/odom')
        self.declare_parameter('path_topic', '/synthetic_path')
        self.declare_parameter('points_topic', '/synthetic_path_points')
        self.declare_parameter('use_points_topic', False)
        self.declare_parameter('hz', 15.0)

        self.declare_parameter('car_length', 0.9)
        self.declare_parameter('car_width', 0.5)
        self.declare_parameter('pad', 2.0)

        # CSV params (match the node that publishes points)
        self.declare_parameter("csv_name", "comp_2021_midpoints.csv")
        self.declare_parameter("csv_file", "")
        self.declare_parameter("columns.mid_x", "mid_x")
        self.declare_parameter("columns.mid_y", "mid_y")

        P = lambda k: self.get_parameter(k).value
        self.odom_topic = str(P('odom_topic'))
        self.path_topic = str(P('path_topic'))
        self.points_topic = str(P('points_topic'))
        self.use_points = bool(P('use_points_topic'))
        self.hz = float(P('hz'))
        self.car_L = float(P('car_length'))
        self.car_W = float(P('car_width'))
        self.view_pad = float(P('pad'))

        csv_name   = str(P("csv_name")).strip()
        csv_file_p = str(P("csv_file")).strip()
        col_x      = str(P("columns.mid_x"))
        col_y      = str(P("columns.mid_y"))

        # -------- state --------
        self.pose: Optional[Tuple[float, float, float]] = None
        self.window_pts: Optional[np.ndarray] = None           # points from topic (left as-is)
        self.base_pts: Optional[np.ndarray] = None             # ALL CSV points (raw)
        self.anchor: Optional[np.ndarray] = None               # first CSV point [x0, y0]

        # -------- load CSV once --------
        try:
            this_dir = FSPath(__file__).resolve().parent
            if csv_file_p:
                csv_path = FSPath(csv_file_p).expanduser().resolve()
                self.get_logger().info(f"Plotter loading CSV (override): {csv_path}")
            else:
                csv_path = (this_dir / "tracks" / csv_name).resolve()
                self.get_logger().info(f"Plotter loading CSV (default): {csv_path}")
            if not csv_path.is_file():
                raise FileNotFoundError(f"CSV not found at: {csv_path}")

            Xs, Ys = self._load_csv_file(csv_path, col_x, col_y)
            self.base_pts = np.stack([Xs, Ys], axis=1)
            if self.base_pts.shape[0] >= 1:
                self.anchor = self.base_pts[0].copy()          # translate CSV so first point -> (0,0)
            self.get_logger().info(f"Plotter loaded {self.base_pts.shape[0]} CSV points for overlay.")
        except Exception as e:
            self.get_logger().error(f"Failed to load CSV overlay: {e}")
            self.base_pts = None
            self.anchor = None

        # -------- QoS: BEST_EFFORT everywhere --------
        qos_best = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=50,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, qos_best)

        if self.use_points:
            self.create_subscription(Float32MultiArray, self.points_topic, self.cb_points, qos_best)
            self.get_logger().info(f'Plotting from Float32MultiArray: {self.points_topic}')
        else:
            self.create_subscription(Path, self.path_topic, self.cb_path, qos_best)
            self.get_logger().info(f'Plotting from nav_msgs/Path: {self.path_topic}')

        # -------- figure --------
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        try:
            self.fig.canvas.manager.set_window_title("Path Window Plotter")
        except Exception:
            pass
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, ls='--', alpha=0.3)
        plt.show(block=False)  # ensure window appears

        # timer
        self.create_timer(max(0.02, 1.0 / max(1e-3, self.hz)), self.on_timer)
        self.get_logger().info('path_window_plotter is live.')

    # ---------- callbacks ----------
    def cb_odom(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.pose = (x, y, yaw)

    def cb_path(self, msg: Path):
        if not msg.poses:
            self.window_pts = None
            return
        xs = [ps.pose.position.x for ps in msg.poses]
        ys = [ps.pose.position.y for ps in msg.poses]
        self.window_pts = np.stack([xs, ys], axis=1)

    def cb_points(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=float)
        if data.size < 2:
            self.window_pts = None
            return
        self.window_pts = data.reshape(-1, 2)

    # ---------- rendering ----------
    def on_timer(self):
        if self.pose is None:
            return

        px, py, yaw = self.pose
        tri = self._car_triangle_world(px, py, yaw)

        self.ax.cla()
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, ls='--', alpha=0.3)
        self.ax.set_title('Car (triangle), CSV track (x, rebased), and Window points (dots)')

        # 1) Draw CSV base track as 'x' markers, TRANSLATED by anchor so first CSV point is (0,0)
        xmin = xmax = px
        ymin = ymax = py
        if self.base_pts is not None and self.base_pts.shape[0] > 0 and self.anchor is not None:
            rebased = self.base_pts - self.anchor[None, :]
            bx, by = rebased[:, 0], rebased[:, 1]
            self.ax.scatter(bx, by, s=20, marker='x', alpha=0.7, label='track CSV (rebased)')
            xmin = min(xmin, float(np.min(bx))); xmax = max(xmax, float(np.max(bx)))
            ymin = min(ymin, float(np.min(by))); ymax = max(ymax, float(np.max(by)))

        # 2) Draw current "window" points as dots (LEFT AS-IS; they may already be rebased by the node)
        if self.window_pts is not None and self.window_pts.shape[0] > 0:
            xs, ys = self.window_pts[:, 0], self.window_pts[:, 1]
            self.ax.plot(xs, ys, linewidth=2.0, alpha=0.8, label='window')
            self.ax.scatter(xs, ys, s=16, alpha=0.9)
            xmin = min(xmin, float(np.min(xs))); xmax = max(xmax, float(np.max(xs)))
            ymin = min(ymin, float(np.min(ys))); ymax = max(ymax, float(np.max(ys)))

        # 3) Draw car triangle (car pose is NOT translated)
        self.ax.fill(tri[:, 0], tri[:, 1], alpha=0.9, label='car')

        # 4) View limits
        xmin = min(xmin, px); xmax = max(xmax, px)
        ymin = min(ymin, py); ymax = max(ymax, py)
        pad = self.view_pad
        if xmax - xmin < 2 * pad:
            cx = 0.5 * (xmin + xmax); xmin, xmax = cx - pad, cx + pad
        if ymax - ymin < 2 * pad:
            cy = 0.5 * (ymin + ymax); ymin, ymax = cy - pad, cy + pad

        self.ax.set_xlim(xmin - pad, xmax + pad)
        self.ax.set_ylim(ymin - pad, ymax + pad)
        self.ax.legend(loc='best')

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _car_triangle_world(self, x: float, y: float, yaw: float) -> np.ndarray:
        L = self.car_L; W = self.car_W
        tri_local = np.array([[+0.5*L, 0.0], [-0.5*L, +0.5*W], [-0.5*L, -0.5*W]], dtype=float)
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        tri_world = tri_local @ R.T
        tri_world[:, 0] += x; tri_world[:, 1] += y
        return tri_world

    # ---------- CSV loader ----------
    def _load_csv_file(self, filepath: FSPath, col_x: str, col_y: str) -> Tuple[np.ndarray, np.ndarray]:
        import csv
        Xs: List[float] = []
        Ys: List[float] = []
        with filepath.open("r", newline="") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames is None:
                raise RuntimeError(f"{filepath}: missing header; need '{col_x},{col_y}'")
            if col_x not in reader.fieldnames or col_y not in reader.fieldnames:
                raise RuntimeError(f"{filepath}: missing columns '{col_x}', '{col_y}'")
            for row in reader:
                Xs.append(float(row[col_x]))
                Ys.append(float(row[col_y]))
        return np.asarray(Xs, dtype=float), np.asarray(Ys, dtype=float)


def main():
    rclpy.init()
    node = PathWindowPlotter()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
