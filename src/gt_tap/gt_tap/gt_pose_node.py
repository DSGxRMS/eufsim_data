#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from gt_tap.gt_utils import quat_to_yaw, norm2

class GTPoseTap(Node):
    def __init__(self):
        super().__init__('gt_pose_tap')
        
        # Subscribe to simulator's ground-truth odometry
        topic_in = '/ground_truth/odom'
        self.sub = self.create_subscription(Odometry, topic_in, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing to: {topic_in}")

        # Publish simplified pose (x, y, yaw)
        topic_out = '/gt_pose'
        self.pub = self.create_publisher(Pose2D, topic_out, 10)
        self.get_logger().info(f"Publishing ground truth pose to: {topic_out}")

    def cb(self, msg: Odometry):
        # Extract position, orientation, and velocity
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear

        # Convert quaternion -> yaw
        yaw = quat_to_yaw(q.w, q.x, q.y, q.z)
        speed_xy = norm2(v.x, v.y)

        # Publish Pose2D (x, y, yaw)
        pose2d = Pose2D()
        pose2d.x = p.x
        pose2d.y = p.y
        pose2d.theta = yaw
        self.pub.publish(pose2d)

        # Optional: log info for debugging
        self.get_logger().info(
            f"x={p.x:.2f}, y={p.y:.2f}, yaw={yaw:.2f}, vx={v.x:.2f}, vy={v.y:.2f}, speed_xy={speed_xy:.2f}"
        )

def main():
    rclpy.init()
    node = GTPoseTap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()