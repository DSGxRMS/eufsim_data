#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class RightFeed(Node):
    def __init__(self):
        super().__init__('right_feed', automatically_declare_parameters_from_overrides=True)
        # Params
        self.declare_parameter('topic', '/zed_right_camera/image_raw')
        self.declare_parameter('reliability', 'best_effort')  # 'best_effort' or 'reliable'
        self.declare_parameter('window', 'right_cam')

        topic = self.get_parameter('topic').get_parameter_value().string_value
        reliability = self.get_parameter('reliability').get_parameter_value().string_value
        window = self.get_parameter('window').get_parameter_value().string_value or 'right_cam'

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT if reliability.lower()=='best_effort'
                        else QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.bridge = CvBridge()
        self.window = window
        self.create_subscription(Image, topic, self.cb, qos)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self.get_logger().info(f'RightFeed: topic={topic} reliability={reliability}')

    def cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imshow(self.window, frame); cv2.waitKey(1)

def main():
    rclpy.init()
    node = RightFeed()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
