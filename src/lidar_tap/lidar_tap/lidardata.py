#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, math, random, csv
import numpy as np

# pandas<2 vs numpy>=1.20 compat shim
if not hasattr(np, "bool"):
    np.bool = bool

import pandas as pd

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2 as pc2
from sklearn.neighbors import KDTree


# -------------------- your helpers (1:1 semantics) --------------------
def euclidean_clustering(P, radius):
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
                if not visited[neighbor_idx]:
                    queue.append(int(neighbor_idx))
                    visited[neighbor_idx] = True
        clusters.append(cluster)
    return clusters


def points_in_cylinder(points, center, axis, radius=0.2, height=1.8):
    d = points - center
    norm = np.linalg.norm(axis)
    if norm == 0:
        norm = 1e-3
    axis = axis / norm
    proj = np.dot(d, axis)
    within_height = (np.abs(proj) <= height / 2)
    proj_vec = np.outer(proj, axis)
    d_perp = (points - center) - proj_vec
    dist_perp = np.linalg.norm(d_perp, axis=1)
    within_radius = dist_perp <= radius
    return within_height & within_radius


def cone_reconstruction(clusters, ground_points, all_points, height=0.8):
    better_clusters = []
    cone_positions = []
    for act_cluster in clusters:
        cluster = ground_points[act_cluster]
        centroid = np.mean(cluster, axis=0)
        cluster = np.array(cluster)
        k = np.argmax(cluster[:, 2])
        top_point = cluster[k]
        axis = top_point - centroid
        if np.linalg.norm(axis) == 0:
            continue
        axis_vec = axis / np.linalg.norm(axis)
        center = top_point + (axis_vec * height / 2)
        mask = points_in_cylinder(all_points, center, axis)
        better_cluster = all_points[mask]
        better_clusters.append(better_cluster)
        position = np.mean(better_cluster, axis=0) if better_cluster.size else centroid
        cone_positions.append(position)
    better_clusters = np.vstack(better_clusters) if len(better_clusters) else np.empty((0, 3))
    return better_clusters, cone_positions


class RANSAC:
    def __init__(self, point_cloud, max_iterations, distance_ratio_threshold):
        self.point_cloud = point_cloud
        self.max_iterations = max_iterations
        self.distance_ratio_threshold = distance_ratio_threshold

    def _ransac_algorithm(self):
        inliers_result = set()
        iters = self.max_iterations
        while iters:
            iters -= 1
            # pick 3 unique
            inliers = []
            seen = set()
            while len(inliers) < 3:
                idx = random.randint(0, len(self.point_cloud) - 1)
                if idx not in seen:
                    inliers.append(idx); seen.add(idx)

            x1, y1, z1 = self.point_cloud.loc[inliers[0]]
            x2, y2, z2 = self.point_cloud.loc[inliers[1]]
            x3, y3, z3 = self.point_cloud.loc[inliers[2]]

            a = (y2 - y1)*(z3 - z1) - (z2 - z1)*(y3 - y1)
            b = (z2 - z1)*(x3 - x1) - (x2 - x1)*(z3 - z1)
            c = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
            d = -(a*x1 + b*y1 + c*z1)
            plane_len = max(0.1, math.sqrt(a*a + b*b + c*c))

            cur = inliers[:]
            for index, row in self.point_cloud.iterrows():
                if index in inliers:
                    continue
                x, y, z = row
                distance = math.fabs(a*x + b*y + c*z + d) / plane_len
                if distance <= self.distance_ratio_threshold:
                    cur.append(index)

            if len(cur) > len(inliers_result):
                inliers_result = set(cur)

        inlier_points = pd.DataFrame(columns=["X", "Y", "Z"])
        outlier_points = pd.DataFrame(columns=["X", "Y", "Z"])
        for index, row in self.point_cloud.iterrows():
            if index in inliers_result:
                inlier_points.loc[len(inlier_points)] = [row["X"], row["Y"], row["Z"]]
            else:
                outlier_points.loc[len(outlier_points)] = [row["X"], row["Y"], row["Z"]]
        return inlier_points, outlier_points


# -------------------- ROS node (no visualization) --------------------
class VelodyneBenchmark(Node):
    def __init__(self):
        super().__init__('pcbench')

        topic = '/velodyne_points'
        self.sub = self.create_subscription(PointCloud2, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing: {topic}")

        # pubs: what you asked for
        self.pub_centroids = self.create_publisher(PointCloud2, '/cones/centroids', 10)
        self.pub_recon     = self.create_publisher(PointCloud2, '/cones/reconstructed_points', 10)

        # knobs (1:1 with your code defaults)
        self.decimate        = 2
        self.ransac_iters    = 25
        self.ransac_thresh   = 0.01
        self.cluster_radius  = 0.5
        self.min_cluster_sz  = None
        self.max_cluster_sz  = None
        self.recon_height    = 0.8

        # timing
        self._frames = 0
        self._last_print = time.time()
        self._ema_total = None

        # CSV logging
        self.write_csv = True
        log_dir = os.path.expanduser('~/eufs_dev/eufs_data/logs')
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, 'cones_benchmark.csv')
        if self.write_csv and not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['stamp', 'n_raw', 't_ransac_ms', 't_cluster_ms', 't_recon_ms', 't_total_ms', 'n_clusters', 'n_cones'])

        self.frame_id = 'velodyne'  # use whatever your sim uses

    def cb(self, msg: PointCloud2):
        # parse cloud -> numpy
        t0 = time.perf_counter()
        gen = pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True)
        P = np.fromiter((c for xyz in gen for c in xyz), dtype=np.float32)
        if P.size == 0:
            return
        P = P.reshape(-1, 3)
        if self.decimate > 1:
            P = P[::int(self.decimate)]
        n_raw = P.shape[0]

        # RANSAC (pandas)
        t1 = time.perf_counter()
        df = pd.DataFrame(P, columns=["X","Y","Z"])
        algo = RANSAC(df, max_iterations=self.ransac_iters, distance_ratio_threshold=self.ransac_thresh)
        ground_df, nonground_df = algo._ransac_algorithm()
        G  = ground_df.to_numpy(dtype=np.float32) if not ground_df.empty else np.empty((0,3), dtype=np.float32)
        NG = nonground_df.to_numpy(dtype=np.float32) if not nonground_df.empty else np.empty((0,3), dtype=np.float32)
        t2 = time.perf_counter()

        # clustering (3D KDTree BFS)
        clusters = euclidean_clustering(NG, self.cluster_radius) if NG.size else []
        if self.min_cluster_sz is not None or self.max_cluster_sz is not None:
            clusters = [c for c in clusters
                        if (self.min_cluster_sz is None or len(c) >= self.min_cluster_sz)
                        and (self.max_cluster_sz is None or len(c) <= self.max_cluster_sz)]
        t3 = time.perf_counter()

        # reconstruction
        if clusters:
            recon_pts, cone_positions = cone_reconstruction(clusters, ground_points=NG, all_points=P, height=self.recon_height)
        else:
            recon_pts = np.empty((0,3), dtype=np.float32)
            cone_positions = []
        t4 = time.perf_counter()

        # publish outputs
        stamp = msg.header.stamp
        if len(cone_positions):
            C = np.vstack(cone_positions).astype(np.float32)
        else:
            C = np.empty((0,3), dtype=np.float32)
        self.pub_centroids.publish(self._to_cloud(C, stamp))
        self.pub_recon.publish(self._to_cloud(recon_pts.astype(np.float32), stamp))

        # timings
        t_ransac = (t2 - t1) * 1000.0
        t_cluster= (t3 - t2) * 1000.0
        t_recon  = (t4 - t3) * 1000.0
        t_total  = (t4 - t0) * 1000.0

        # smooth FPS
        if self._ema_total is None:
            self._ema_total = t_total
        else:
            self._ema_total = 0.9*self._ema_total + 0.1*t_total
        fps = 1000.0 / max(1e-3, self._ema_total)

        # print occasionally
        self._frames += 1
        now = time.time()
        if now - self._last_print >= 0.5:
            self._last_print = now
            self.get_logger().info(
                f"raw:{n_raw:6d}  R:{t_ransac:6.1f}ms  Cl:{t_cluster:6.1f}ms  Re:{t_recon:6.1f}ms  Tot:{t_total:6.1f}ms  "
                f"clusters:{len(clusters):3d}  cones:{len(cone_positions):3d}  ~FPS:{fps:4.1f}"
            )

        # CSV log
        if self.write_csv:
            with open(self.csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([stamp.sec + stamp.nanosec*1e-9, n_raw, f"{t_ransac:.3f}", f"{t_cluster:.3f}",
                            f"{t_recon:.3f}", f"{t_total:.3f}", len(clusters), len(cone_positions)])

    # --------- helpers ---------
    def _to_cloud(self, pts: np.ndarray, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        if pts.size == 0:
            data = np.zeros((0,3), dtype=np.float32)
        else:
            data = np.ascontiguousarray(pts.astype(np.float32))
        return pc2.create_cloud(header, fields, data)
    

def main():
    rclpy.init()
    node = VelodyneBenchmark()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
