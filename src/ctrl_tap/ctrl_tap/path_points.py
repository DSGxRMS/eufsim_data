#!/usr/bin/env python3
# path_window_node.py
#
# Rebuild path to start at CSV point nearest (0,0) and go to the end (no wrap).
# Runtime:
#   - Forward-only latch along CSV (never goes backward).
#   - Publish: [CAR POSE] + next_k CSV points (indices >= current).
#   - Yaw calibration: wait for car to move >= calib_dist from first pose,
#     then set calibrated_yaw = atan2(dy, dx) based on that motion.
#
import math
from typing import Tuple, Optional, List
from pathlib import Path as FSPath

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

from nav_msgs.msg import Odometry, Path as RosPath
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker


class PathWindowNode(Node):
    def __init__(self):
        super().__init__("path_window_node", automatically_declare_parameters_from_overrides=True)

        # ---------------- Params ----------------
        self.declare_parameter("csv_name", "comp_2021_midpoints.csv")
        self.declare_parameter("csv_file", "")
        self.declare_parameter("columns.mid_x", "mid_x")
        self.declare_parameter("columns.mid_y", "mid_y")

        # publish rate and window size
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("next_k", 4)              # car pose + next_k CSV points
        self.declare_parameter("search_stride", 1)       # stride for NN on rebuilt list

        # yaw calibration
        self.declare_parameter("calib_dist", 0.4)        # meters to move before locking yaw from motion

        # Topics
        self.declare_parameter("odom_topic", "/ground_truth/odom")
        self.declare_parameter("path_topic", "/synthetic_path")
        self.declare_parameter("points_topic", "/synthetic_path_points")
        self.declare_parameter("marker_topic", "/synthetic_path_marker")

        # --------------- Fetch params ---------------
        P = lambda k: self.get_parameter(k).value
        csv_name       = str(P("csv_name")).strip()
        csv_file_p     = str(P("csv_file")).strip()
        col_x          = str(P("columns.mid_x"))
        col_y          = str(P("columns.mid_y"))

        self.publish_hz     = float(P("publish_hz"))
        self.NEXT_K         = max(0, int(P("next_k")))
        self.search_stride  = max(1, int(P("search_stride")))
        self.calib_dist     = float(P("calib_dist"))

        self.odom_topic   = str(P("odom_topic"))
        self.path_topic   = str(P("path_topic"))
        self.points_topic = str(P("points_topic"))
        self.marker_topic = str(P("marker_topic"))

        # --------------- Resolve and load path CSV ---------------
        this_dir = FSPath(__file__).resolve().parent
        if csv_file_p:
            csv_path = FSPath(csv_file_p).expanduser().resolve()
            self.get_logger().info(f"Loading CSV (override): {csv_path}")
        else:
            csv_path = (this_dir / "tracks" / csv_name).resolve()
            self.get_logger().info(f"Loading CSV (default): {csv_path}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found at: {csv_path}")

        X, Y = self._load_csv_file(csv_path, col_x, col_y)
        if len(X) < 2:
            raise RuntimeError("Not enough points loaded from CSV. Need >= 2.")

        pts_orig = np.stack([X, Y], axis=1)

        # Rebuild: start at point nearest to (0,0), then go to the end (no wrap)
        i0 = int(np.argmin(np.einsum("ij,ij->i", pts_orig, pts_orig)))
        self.PTS = pts_orig[i0:]
        self.M   = self.PTS.shape[0]
        self.get_logger().info(f"Rebuilt path: start index {i0} (nearest to 0,0), total {self.M} points.")

        # --------------- IO (BEST_EFFORT everywhere) ---------------
        qos_best = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=50,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, qos_best)
        self.pub_path   = self.create_publisher(RosPath,           self.path_topic,   qos_best)
        self.pub_points = self.create_publisher(Float32MultiArray, self.points_topic, qos_best)
        self.pub_marker = self.create_publisher(Marker,            self.marker_topic, qos_best)

        # runtime state
        self.last_odom: Optional[Odometry] = None
        self.curr_idx: Optional[int] = None  # monotonic index along rebuilt list

        # yaw calibration state
        self.first_pose: Optional[np.ndarray] = None   # (x0, y0)
        self.calibrated_yaw: Optional[float] = None

        self.timer = self.create_timer(max(0.01, 1.0 / max(1e-3, self.publish_hz)), self.on_timer)

    # ---------------- CSV loading ----------------
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
                Xs.append(float(row[col_x])); Ys.append(float(row[col_y]))
        return np.asarray(Xs, dtype=float), np.asarray(Ys, dtype=float)

    # ---------------- Odometry callback ----------------
    def cb_odom(self, msg: Odometry):
        self.last_odom = msg  # untouched

    # ---------------- Timer ----------------
    def on_timer(self):
        if self.last_odom is None or self.M == 0:
            return

        msg = self.last_odom
        px = float(msg.pose.pose.position.x)
        py = float(msg.pose.pose.position.y)
        pos = np.array([px, py], dtype=float)

        # --- yaw calibration from motion ---
        if self.first_pose is None:
            self.first_pose = pos.copy()
        elif self.calibrated_yaw is None:
            dx = px - self.first_pose[0]
            dy = py - self.first_pose[1]
            if (dx*dx + dy*dy) >= (self.calib_dist * self.calib_dist):
                self.calibrated_yaw = math.atan2(dy, dx)
                self.get_logger().info(f"Calibrated yaw from motion: {self.calibrated_yaw:.3f} rad")

        # --- forward-only nearest index ---
        nn_idx = self._nearest_index_forward_only(pos)
        if self.curr_idx is None:
            self.curr_idx = nn_idx
        else:
            self.curr_idx = max(self.curr_idx, nn_idx)  # never go backwards

        # Build window: [CAR] + next_k CSV points with index >= curr_idx
        idx_start = self.curr_idx
        idx_end   = min(self.M, idx_start + self.NEXT_K)
        csv_indices = np.arange(idx_start, idx_end, dtype=int)

        frame_id = msg.header.frame_id or "map"
        self._publish_path_with_car(csv_indices, msg.header.stamp, frame_id, px, py)
        self._publish_points_with_car(csv_indices, px, py)
        self._publish_marker_with_car(csv_indices, frame_id, px, py)

    # ---------------- Helpers ----------------
    def _nearest_index_forward_only(self, pos: np.ndarray) -> int:
        """Euclidean NN on rebuilt list; if we have curr_idx, only consider indices >= curr_idx."""
        if self.curr_idx is None:
            cand_idx = np.arange(0, self.M, self.search_stride, dtype=int)
        else:
            cand_idx = np.arange(self.curr_idx, self.M, self.search_stride, dtype=int)
            if cand_idx.size == 0:
                return self.M - 1
        cand_pts = self.PTS[cand_idx]
        d2 = np.einsum('ij,ij->i', cand_pts - pos, cand_pts - pos)
        return int(cand_idx[int(np.argmin(d2))])

    # ---------------- Publishers ----------------
    def _publish_path_with_car(self, idxs: np.ndarray, stamp, frame_id: str, px: float, py: float):
        path = RosPath()
        path.header.stamp = stamp
        path.header.frame_id = frame_id

        # 1) Car as first pose, orientation from calibrated yaw when available
        ps_car = PoseStamped()
        ps_car.header.stamp = stamp
        ps_car.header.frame_id = frame_id
        ps_car.pose.position.x = px
        ps_car.pose.position.y = py
        ps_car.pose.position.z = 0.0
        if self.calibrated_yaw is not None:
            half = 0.5 * self.calibrated_yaw
            ps_car.pose.orientation.z = math.sin(half)
            ps_car.pose.orientation.w = math.cos(half)
        else:
            # not calibrated yet -> identity orientation
            ps_car.pose.orientation.w = 1.0
        path.poses.append(ps_car)

        # 2) Then CSV points
        for i in idxs:
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = frame_id
            ps.pose.position.x = float(self.PTS[i, 0])
            ps.pose.position.y = float(self.PTS[i, 1])
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)

    def _publish_points_with_car(self, idxs: np.ndarray, px: float, py: float):
        arr = Float32MultiArray()
        out = [float(px), float(py)]
        for i in idxs:
            out.extend([float(self.PTS[i, 0]), float(self.PTS[i, 1])])
        arr.data = out
        self.pub_points.publish(arr)

    def _publish_marker_with_car(self, idxs: np.ndarray, frame_id: str, px: float, py: float):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "synthetic_path"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.a = 1.0
        m.color.r = 0.1
        m.color.g = 1.0
        m.color.b = 0.1

        from geometry_msgs.msg import Point
        pcar = Point(); pcar.x = float(px); pcar.y = float(py); pcar.z = 0.0
        m.points.append(pcar)
        for i in idxs:
            p = Point()
            p.x = float(self.PTS[i, 0]); p.y = float(self.PTS[i, 1]); p.z = 0.0
            m.points.append(p)
        self.pub_marker.publish(m)


def main():
    rclpy.init()
    node = PathWindowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
