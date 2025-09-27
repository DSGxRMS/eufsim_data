#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import time

class ImageProbe(Node):
    def __init__(self):
        super().__init__('image_probe', automatically_declare_parameters_from_overrides=True)
        self.declare_parameter('topic', '/zed/left/image_rect_color')
        self.declare_parameter('compressed', False)

        topic = self.get_parameter('topic').get_parameter_value().string_value
        compressed = self.get_parameter('compressed').get_parameter_value().bool_value

        self.count = 0
        self.t0 = time.time()
        self.bytes = 0

        if compressed:
            self.create_subscription(CompressedImage, topic, self.cb_compressed, 10)
        else:
            self.create_subscription(Image, topic, self.cb_raw, 10)

        self.get_logger().info(f'Probing topic: {topic} (compressed={compressed})')

    def cb_raw(self, msg: Image):
        self.count += 1
        self.bytes += len(msg.data)
        self._maybe_log(msg.header.stamp.sec, msg.width, msg.height)

    def cb_compressed(self, msg: CompressedImage):
        self.count += 1
        self.bytes += len(msg.data)
        self._maybe_log(msg.header.stamp.sec, -1, -1)

    def _maybe_log(self, sec, w, h):
        now = time.time()
        if now - self.t0 >= 2.0:
            hz = self.count / (now - self.t0)
            mbps = (self.bytes * 8.0) / (now - self.t0) / 1e6
            wh = f'{w}x{h}' if w > 0 else 'compressed'
            self.get_logger().info(f'frames={self.count}  ~{hz:.1f} Hz  ~{mbps:.2f} Mbit/s  ({wh})')
            self.t0 = now
            self.count = 0
            self.bytes = 0

def main():
    rclpy.init()
    node = ImageProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
