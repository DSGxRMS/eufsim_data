# lidar_tap/pointcloud_velodyne.py
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

class VelodyneTap(Node):
    def __init__(self):
        super().__init__('velodyne_tap')
        topic = '/velodyne_points'  # fixed to velodyne (can be changed accordingly)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.sub = self.create_subscription(PointCloud2, topic, self.cb, qos)
        self.last_print = time.time()
        self.frames = 0
        self.points_seen = 0
        self.get_logger().info(f"Listening to PointCloud2 on: {topic}")

    def cb(self, msg: PointCloud2):
        # Read a subset so we don't burn CPU
        gen = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        sample = []
        for i, p in enumerate(gen):
            self.points_seen += 1
            if i < 5:
                sample.append((float(p[0]), float(p[1]), float(p[2])))
            if i >= 5000:
                break
        self.frames += 1
        now = time.time()
        if now - self.last_print > 1.0:
            self.get_logger().info(
                f"frames/s={self.frames} points≈{self.points_seen} sample5={sample}"
            )
            self.last_print = now
            self.frames = 0
            self.points_seen = 0

def main():
    rclpy.init()
    node = VelodyneTap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
