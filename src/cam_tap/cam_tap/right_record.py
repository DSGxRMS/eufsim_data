#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pathlib import Path
from datetime import datetime

class LeftFeedRecorder(Node):
    def __init__(self):
        super().__init__('left_feed_recorder', automatically_declare_parameters_from_overrides=True)

        # ---- Params (compatible with EUFSIM-style config via CLI/params) ----
        self.declare_parameter('topic', '/zed/right/image_rect_color')
        self.declare_parameter('reliability', 'best_effort')       # 'best_effort' | 'reliable'
        self.declare_parameter('window', 'left_cam')               # used for default filename prefix
        self.declare_parameter('preview', False)                   # True to also show a preview window
        self.declare_parameter('fps', 30.0)                        # recording FPS
        self.declare_parameter('codec', 'mp4v')                    # 'mp4v' is widely supported for .mp4
        self.declare_parameter('output_dir', 'data')               # folder relative to this script
        self.declare_parameter('filename_prefix', '')              # override default prefix if you want

        topic = self.get_parameter('topic').get_parameter_value().string_value
        reliability = self.get_parameter('reliability').get_parameter_value().string_value
        window = self.get_parameter('window').get_parameter_value().string_value or 'left_cam'
        preview = self.get_parameter('preview').get_parameter_value().bool_value
        fps = float(self.get_parameter('fps').get_parameter_value().double_value)
        codec = self.get_parameter('codec').get_parameter_value().string_value or 'mp4v'
        output_dir_name = self.get_parameter('output_dir').get_parameter_value().string_value or 'data'
        filename_prefix_param = self.get_parameter('filename_prefix').get_parameter_value().string_value

        qos = QoSProfile(
            reliability=(QoSReliabilityPolicy.BEST_EFFORT
                         if reliability.lower() == 'best_effort'
                         else QoSReliabilityPolicy.RELIABLE),
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ---- Paths & state ----
        # data folder placed alongside this file, so it works no matter where you ros2 run from
        script_dir = Path(__file__).resolve().parent
        self.output_dir = script_dir / output_dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build a filename (prefix from param -> else window name -> else topic tail)
        if filename_prefix_param.strip():
            prefix = filename_prefix_param.strip()
        else:
            # sanitize topic for filename if needed
            topic_tail = topic.strip('/').replace('/', '_') or 'camera'
            prefix = window or topic_tail

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out_path = self.output_dir / f'{prefix}_{ts}.mp4'

        self.bridge = CvBridge()
        self.preview = preview
        self.window = window

        # Video writer created lazily on first frame (once we know width/height)
        self.writer = None
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*codec)

        self.sub = self.create_subscription(Image, topic, self.cb, qos)
        if self.preview:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f'Recorder started: topic="{topic}", reliability="{reliability}", '
            f'fps={self.fps}, codec="{codec}", saving to: {self.out_path}'
        )

    def _init_writer_if_needed(self, frame):
        if self.writer is not None:
            return
        h, w = frame.shape[:2]
        self.writer = cv2.VideoWriter(str(self.out_path), self.fourcc, self.fps, (w, h))
        if not self.writer.isOpened():
            self.get_logger().error(
                f'Failed to open VideoWriter for "{self.out_path}". '
                f'Check codec/backends (tried fourcc="{chr(self.fourcc & 0xff)}'
                f'{chr((self.fourcc>>8) & 0xff)}{chr((self.fourcc>>16) & 0xff)}'
                f'{chr((self.fourcc>>24) & 0xff)}").'
            )
        else:
            self.get_logger().info(f'VideoWriter initialized: {w}x{h}@{self.fps} -> {self.out_path}')

    def cb(self, msg: Image):
        # Convert to BGR for OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'CvBridge conversion failed: {e}')
            return

        # Lazy-init writer with actual frame size
        self._init_writer_if_needed(frame)

        # Write frame if writer ok
        if self.writer is not None and self.writer.isOpened():
            self.writer.write(frame)

        # Optional preview
        if self.preview:
            cv2.imshow(self.window, frame)
            cv2.waitKey(1)

    def destroy_node(self):
        # Clean shutdown
        try:
            if self.writer is not None:
                self.writer.release()
        except Exception:
            pass
        if self.preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        super().destroy_node()

def main():
    rclpy.init()
    node = LeftFeedRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
