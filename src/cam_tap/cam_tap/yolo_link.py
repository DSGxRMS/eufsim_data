#!/usr/bin/env python3
import time
import numpy as np
import cv2
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def letterbox(im_bgr, new_shape=736, color=(114, 114, 114)):
    """Resize + pad to square while preserving aspect ratio."""
    h, w = im_bgr.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        im_bgr = cv2.resize(im_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_bgr = cv2.copyMakeBorder(im_bgr, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return im_bgr, r, (left, top)


def nms_xyxy(boxes, scores, iou_thresh=0.7):
    """Pure numpy NMS. boxes Nx4 (xyxy), scores N."""
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


class YoloOnnxRunner:
    """
    Runs Ultralytics YOLO ONNX export with ONNXRuntime.
    Assumes output format is [x,y,w,h, class_scores...].
    Handles common output shapes: (1,N,4+nc) or (1,4+nc,N).
    """
    def __init__(self, onnx_path, imgsz=736, conf=0.25, iou=0.7, class_names=None, use_cuda=True):
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.class_names = class_names or ["yellow", "blue", "orange", "large_orange"]
        self.nc = len(self.class_names)

        providers = ["CPUExecutionProvider"]
        if use_cuda:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

        print("[YOLO ONNX] Providers active:", self.sess.get_providers())
        # Warmup
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        _ = self.infer(dummy)

    def infer(self, bgr):
        """Returns list of detections: (cls_id, score, x1,y1,x2,y2) in ORIGINAL image coords."""
        orig_h, orig_w = bgr.shape[:2]

        lb, r, (padx, pady) = letterbox(bgr, self.imgsz)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)

        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3xHxW

        y = self.sess.run([self.output_name], {self.input_name: x})[0]
        y = np.asarray(y)

        if y.ndim == 3:
            y = y[0]

        # Normalize shape to (N, 4+nc)
        if y.shape[0] == (4 + self.nc) and y.shape[1] > 10:
            y = y.transpose(1, 0)
        elif y.shape[-1] == (4 + self.nc):
            pass
        else:
            raise RuntimeError(f"Unexpected ONNX output shape: {y.shape}. "
                               f"Expected (N, {4+self.nc}) or ({4+self.nc}, N).")

        boxes_xywh = y[:, :4]
        cls_scores = y[:, 4:4 + self.nc]

        cls_id = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_id]

        keep = scores >= self.conf
        boxes_xywh = boxes_xywh[keep]
        cls_id = cls_id[keep]
        scores = scores[keep]

        if boxes_xywh.shape[0] == 0:
            return []

        # xywh -> xyxy in letterbox coords
        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # unletterbox to original coords
        boxes[:, [0, 2]] -= padx
        boxes[:, [1, 3]] -= pady
        boxes /= r

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        # per-class NMS
        dets = []
        for c in range(self.nc):
            idx = np.where(cls_id == c)[0]
            if idx.size == 0:
                continue
            b = boxes[idx]
            s = scores[idx]
            k = nms_xyxy(b, s, self.iou)
            for j in k:
                x1, y1, x2, y2 = b[j]
                dets.append((int(c), float(s[j]), int(x1), int(y1), int(x2), int(y2)))

        return dets

from pathlib import Path
YOLOPATH = Path(__file__).parent / "yolov11" / "weights" / "best.onnx"
class YoloLiveViewer(Node):
    def __init__(self):
        super().__init__("yolo_live_viewer")
        self.bridge = CvBridge()

        # Params
        self.declare_parameter("image_topic", "/zed/left/image_rect_color")
        self.declare_parameter("model_path", YOLOPATH.as_posix())
        self.declare_parameter("imgsz", 736)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("use_cuda", True)
        self.declare_parameter("window_name", "YOLO Live")
        self.declare_parameter("class_names", ["yellow", "blue", "orange", "large_orange"])

        self.image_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.use_cuda = bool(self.get_parameter("use_cuda").value)
        self.window_name = self.get_parameter("window_name").value
        self.class_names = list(self.get_parameter("class_names").value)

        self.get_logger().info(f"Subscribing: {self.image_topic}")
        self.get_logger().info(f"Loading ONNX: {self.model_path} (CUDA={self.use_cuda})")

        self.yolo = YoloOnnxRunner(
            onnx_path=self.model_path,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            class_names=self.class_names,
            use_cuda=self.use_cuda
        )

        self.sub = self.create_subscription(
            Image, self.image_topic, self.on_image, qos_profile_sensor_data
        )

        # FPS tracking
        self.last_t = time.time()
        self.fps = 0.0

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def on_image(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        t0 = time.time()
        dets = self.yolo.infer(bgr)
        infer_ms = (time.time() - t0) * 1000.0

        # FPS (EMA)
        now = time.time()
        dt = now - self.last_t
        self.last_t = now
        if dt > 1e-6:
            inst_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * inst_fps if self.fps > 0 else inst_fps

        # Draw detections
        for (cls_id, score, x1, y1, x2, y2) in dets:
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{self.class_names[cls_id]} {score:.2f}"
            cv2.putText(bgr, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Overlay perf
        cv2.putText(bgr, f"FPS: {self.fps:.1f}  infer: {infer_ms:.1f} ms  dets: {len(dets)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(self.window_name, bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit requested (q). Shutting down.")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = YoloLiveViewer()
    try:
        rclpy.spin(node)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
