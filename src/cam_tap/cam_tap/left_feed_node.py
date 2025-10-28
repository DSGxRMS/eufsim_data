#!/usr/bin/env python3
# right_feed_yolo_overlay.py
#
# Live camera -> YOLO -> annotated overlay
# - Subscribes to raw or compressed images
# - Overlays bboxes + labels
# - Publishes annotated stream on a ROS2 topic (raw or compressed)
# - Optional on-screen preview window
#
# Params (with sensible defaults):
#   topic:                zed/right/image_rect_color
#   reliability:          best_effort | reliable
#   in_transport:         raw | compressed
#   out_transport:        raw | compressed
#   out_topic:            /right/overlay
#   window:               right_cam_overlay
#   show_window:          true | false
#   max_fps:              12.0   (drop frames if faster)
#   yolo_repo_rel:        yolov5
#   yolo_weights_rel:     yolov5/weights/best.pt
#   yolo_device:          cuda | cpu
#   conf_thres:           0.75
#   iou_thres:            0.35
#
# Notes:
# - Expect RGB input to YOLO; convert back to BGR for drawing/imshow/encoded publish.
# - If GPU is not available, it auto-falls back to CPU.

import io
import math
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CompressedImage as RosCompressedImage
from cv_bridge import CvBridge

# ─────────────────────────────────────────────────────────────────────────────
# YOLOv5 wrapper (condensed from your reference; same API: infer(), draw_rgb(), dets_to_yolo_txt())
# ─────────────────────────────────────────────────────────────────────────────
class YOLOv5Detector:
    def __init__(self, repo_dir: Path, weights_path: Path, device: str = "cuda",
                 conf_thres: float = 0.75, iou_thres: float = 0.35, input_size: int = 640):
        import sys
        import torch

        self.repo_dir = Path(repo_dir).resolve()
        self.weights_path = Path(weights_path).resolve()
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        self.backend = None
        self.model = None
        self.names = {}
        self.input_size = int(input_size)
        self.conf_thres = float(conf_thres)
        self.iou_thres  = float(iou_thres)

        # Try 1: torch.hub local
        try:
            self.model = torch.hub.load(
                repo_or_dir=str(self.repo_dir),
                model='custom',
                path=str(self.weights_path),
                source='local'
            ).to(self.device)
            self.model.eval()
            self.names = self.model.names if hasattr(self.model, "names") else {0: "obj"}
            self.backend = "hub"
            return
        except Exception:
            pass

        # Try 2: direct local imports
        try:
            sys.path.insert(0, str(self.repo_dir))
            from models.common import DetectMultiBackend
            from utils.augmentations import letterbox as _letterbox
            from utils.general import non_max_suppression as _nms, scale_boxes as _scale

            self.DetectMultiBackend = DetectMultiBackend
            self.letterbox = _letterbox
            self.nms = _nms
            self.scale_boxes = _scale

            self.model = self.DetectMultiBackend(str(self.weights_path), device=self.device, dnn=False, data=None, fp16=False)
            self.names = getattr(self.model, "names", None) or {0: "obj"}
            self.backend = "raw"
            return
        except Exception as e:
            raise RuntimeError(
                f"YOLOv5 load failed. repo={self.repo_dir} weights={self.weights_path}. Error: {e}"
            )

    def infer(self, img_rgb: np.ndarray):
        if self.backend == "hub":
            pil = PILImage.fromarray(img_rgb)
            results = self.model(pil, size=self.input_size)
            dets = []
            if hasattr(results, "xyxy") and len(results.xyxy) > 0:
                preds = results.xyxy[0].detach().cpu().numpy()
                for x1, y1, x2, y2, conf, cls in preds:
                    if float(conf) < self.conf_thres:
                        continue
                    dets.append({
                        "x1": int(round(x1)), "y1": int(round(y1)),
                        "x2": int(round(x2)), "y2": int(round(y2)),
                        "conf": float(conf), "cls": int(cls),
                        "name": self.names[int(cls)] if int(cls) in self.names else str(int(cls)),
                    })
            return dets

        elif self.backend == "raw":
            im = self.letterbox(img_rgb, new_shape=self.input_size, stride=32, auto=True)[0]
            im = im.transpose((2, 0, 1))  # HWC->CHW
            im = np.ascontiguousarray(im)
            im = self.torch.from_numpy(im).to(self.device).float() / 255.0
            if im.ndim == 3:
                im = im.unsqueeze(0)

            pred = self.model(im)
            pred = self.nms(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]
            out = []
            if pred is None or len(pred) == 0:
                return out

            pred[:, :4] = self.scale_boxes(im.shape[2:], pred[:, :4], img_rgb.shape).round()
            for *xyxy, conf, cls in pred.detach().cpu().tolist():
                x1, y1, x2, y2 = map(int, xyxy)
                cls = int(cls)
                out.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": float(conf), "cls": cls,
                    "name": self.names[cls] if cls in self.names else str(cls),
                })
            return out

        else:
            raise RuntimeError("YOLO backend not initialized")

    @staticmethod
    def draw_rgb(img_rgb: np.ndarray, dets):
        for d in dets:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            label = f'{d["name"]} {d["conf"]:.2f}'
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_rgb, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(img_rgb, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img_rgb


# ─────────────────────────────────────────────────────────────────────────────
# Live overlay node
# ─────────────────────────────────────────────────────────────────────────────
class RightFeedYOLO(Node):
    def __init__(self):
        super().__init__('right_feed_yolo', automatically_declare_parameters_from_overrides=True)

        # Params
        self.declare_parameter('topic', 'zed/left/image_rect_color')
        self.declare_parameter('reliability', 'best_effort')              # 'best_effort' or 'reliable'
        self.declare_parameter('in_transport', 'raw')                     # 'raw' or 'compressed'
        self.declare_parameter('out_transport', 'raw')                    # 'raw' or 'compressed'
        self.declare_parameter('out_topic', '/right/overlay')
        self.declare_parameter('window', 'right_cam_overlay')
        self.declare_parameter('show_window', True)
        self.declare_parameter('max_fps', 12.0)

        # YOLO params
        self.declare_parameter('yolo_repo_rel', 'yolov5')
        self.declare_parameter('yolo_weights_rel', 'yolov5/weights/best.pt')
        self.declare_parameter('yolo_device', 'cuda')
        self.declare_parameter('conf_thres', 0.75)
        self.declare_parameter('iou_thres', 0.35)

        gp = self.get_parameter
        topic         = gp('topic').get_parameter_value().string_value
        reliability   = gp('reliability').get_parameter_value().string_value.lower()
        in_transport  = gp('in_transport').get_parameter_value().string_value.lower()
        out_transport = gp('out_transport').get_parameter_value().string_value.lower()
        out_topic     = gp('out_topic').get_parameter_value().string_value
        self.window   = gp('window').get_parameter_value().string_value or 'right_cam_overlay'
        self.show_win = gp('show_window').get_parameter_value().bool_value
        self.max_fps  = float(gp('max_fps').get_parameter_value().double_value)

        # YOLO config
        base = Path(__file__).parent
        repo_dir   = (base / gp('yolo_repo_rel').get_parameter_value().string_value).resolve()
        weights_pt = (base / gp('yolo_weights_rel').get_parameter_value().string_value).resolve()
        y_dev      = gp('yolo_device').get_parameter_value().string_value
        conf_thr   = float(gp('conf_thres').get_parameter_value().double_value)
        iou_thr    = float(gp('iou_thres').get_parameter_value().double_value)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT if reliability == 'best_effort'
                        else QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.bridge = CvBridge()
        self.in_transport = in_transport
        self.out_transport = out_transport

        # Publishers
        if self.out_transport == 'compressed':
            self.pub = self.create_publisher(RosCompressedImage, out_topic, qos)
        else:
            self.pub = self.create_publisher(RosImage, out_topic, qos)

        # Subscriber
        if self.in_transport == 'compressed':
            self.sub = self.create_subscription(RosCompressedImage, topic, self._cb_compressed, qos)
            self.get_logger().info(f'[img] Subscribed (Compressed) to: {topic}')
        else:
            self.sub = self.create_subscription(RosImage, topic, self._cb_raw, qos)
            self.get_logger().info(f'[img] Subscribed (Raw) to: {topic}')

        # Window
        if self.show_win:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        # YOLO init
        self.get_logger().info(f'[yolo] repo={repo_dir} weights={weights_pt} device={y_dev} '
                               f'conf={conf_thr} iou={iou_thr}')
        self.yolo = YOLOv5Detector(repo_dir=repo_dir, weights_path=weights_pt,
                                   device=y_dev, conf_thres=conf_thr, iou_thres=iou_thr)

        # Frame buffer (latest only) + timing
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_stamp = None
        self._last_run_t = 0.0
        self._frame_drop = 0

        # Timer for throttled processing
        tick_period = 1.0 / max(1.0, self.max_fps)
        self.timer = self.create_timer(tick_period, self._tick)

        self.get_logger().info(f'RightFeedYOLO: topic={topic}, out_topic={out_topic}, '
                               f'in={self.in_transport}, out={self.out_transport}, '
                               f'reliability={reliability}, max_fps={self.max_fps}')

    # ─────────── Sub callbacks: keep only latest frame ───────────
    def _cb_raw(self, msg: RosImage):
        rgb = self._rgb_from_raw(msg)
        if rgb is None:
            return
        self._latest_rgb = rgb
        self._latest_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)

    def _cb_compressed(self, msg: RosCompressedImage):
        rgb = self._rgb_from_compressed(msg)
        if rgb is None:
            return
        self._latest_rgb = rgb
        self._latest_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)

    # ─────────── Throttled processing/publish ───────────
    def _tick(self):
        now = time.time()
        period = 1.0 / max(1.0, self.max_fps)
        # If no new frame or too soon, skip
        if self._latest_rgb is None or (now - self._last_run_t) < period:
            return

        rgb = self._latest_rgb
        self._last_run_t = now

        # Run YOLO on RGB
        dets = self.yolo.infer(rgb)

        # Draw on a copy in RGB, then convert to BGR for display/publish
        ann_rgb = self.yolo.draw_rgb(rgb.copy(), dets)
        ann_bgr = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)

        # On-screen preview
        if self.show_win:
            cv2.imshow(self.window, ann_bgr)
            cv2.waitKey(1)

        # Publish
        if self.out_transport == 'compressed':
            # JPEG encode
            ok, enc = cv2.imencode('.jpg', ann_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return
            msg_out = RosCompressedImage()
            msg_out.format = 'jpeg'
            msg_out.data = enc.tobytes()
            self.pub.publish(msg_out)
        else:
            # raw bgr8
            self.pub.publish(self.bridge.cv2_to_imgmsg(ann_bgr, encoding='bgr8'))

    # ─────────── Converters ───────────
    def _rgb_from_raw(self, msg: RosImage) -> Optional[np.ndarray]:
        # Use CvBridge for robustness; force to bgr8, then convert
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().warn(f'raw convert failed: {e}')
            return None

    def _rgb_from_compressed(self, msg: RosCompressedImage) -> Optional[np.ndarray]:
        try:
            bio = io.BytesIO(bytes(msg.data))
            pil = PILImage.open(bio).convert('RGB')
            return np.array(pil, dtype=np.uint8)
        except Exception as e:
            self.get_logger().warn(f'compressed convert failed: {e}')
            return None


def main():
    rclpy.init()
    node = RightFeedYOLO()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node.show_win:
                cv2.destroyAllWindows()
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
