#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as RosImage, CompressedImage as RosCompressedImage
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, qos_profile_sensor_data
)

from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray


class YOLOv5Detector:
    """
    Robust loader:
      - First try torch.hub.load(local_repo, 'custom', path=weights, source='local')
      - Fallback: sys.path.insert(0, repo) + import models/common, utils/*
      - Same public API: infer(), draw_rgb(), dets_to_yolo_txt()
    """
    def __init__(self, repo_dir: Path, weights_path: Path, device: str = "cuda"):
        import os, sys
        import torch

        self.repo_dir = Path(repo_dir).resolve()
        self.weights_path = Path(weights_path).resolve()

        # Debug prints so you can SEE exactly what it's using
        print("[yolo] torch.__version__ =", torch.__version__)
        print("[yolo] sys.executable    =", sys.executable)
        print("[yolo] repo_dir          =", self.repo_dir, " exists:", self.repo_dir.exists())
        print("[yolo] weights_path      =", self.weights_path, " exists:", self.weights_path.exists())
        print("[yolo] hubconf.py exists =", (self.repo_dir / "hubconf.py").exists())

        self.torch = torch
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

        self.backend = None
        self.model = None
        self.names = {}
        self.input_size = 640
        self.conf_thres = 0.75
        self.iou_thres  = 0.35

        # Try 1: Torch Hub local
        try:
            print("[yolo] Attempting torch.hub.load(source='local')...")
            self.model = torch.hub.load(
                repo_or_dir=str(self.repo_dir),
                model='custom',
                path=str(self.weights_path),
                source='local'
            ).to(self.device)
            self.model.eval()
            self.names = self.model.names if hasattr(self.model, "names") else {0: "obj"}
            self.backend = "hub"
            print("[yolo] Loaded via torch.hub (local).")
            return
        except Exception as e:
            print("[yolo] torch.hub local load FAILED:", repr(e))

        # Try 2: Direct local imports from repo path (no 'import yolov5')
        try:
            print("[yolo] Attempting direct local imports from repo path...")
            sys.path.insert(0, str(self.repo_dir))

            # In-repo import style (e.g., models/common.py, utils/*)
            from models.common import DetectMultiBackend
            from utils.augmentations import letterbox as _letterbox
            from utils.general import non_max_suppression as _nms, scale_boxes as _scale

            self.DetectMultiBackend = DetectMultiBackend
            self.letterbox = _letterbox
            self.nms = _nms
            self.scale_boxes = _scale

            self.model = self.DetectMultiBackend(
                str(self.weights_path),
                device=self.device,
                dnn=False,
                data=None,
                fp16=False
            )
            # class names
            self.names = getattr(self.model, "names", None) or {0: "obj"}
            self.backend = "raw"
            print("[yolo] Loaded via direct local imports.")
            return
        except Exception as e:
            print("[yolo] direct local import FAILED:", repr(e))

        # If we get here, both approaches failed
        raise RuntimeError(
            "Could not load YOLOv5. Check that:\n"
            f" - repo_dir exists and has hubconf.py: {self.repo_dir}\n"
            f" - weights_path exists: {self.weights_path}\n"
            " - dependencies from yolov5/requirements.txt are installed in THIS Python env\n"
        )

    def infer(self, img_rgb: np.ndarray):
        """
        Run YOLOv5 and return list of dicts: {x1,y1,x2,y2,conf,cls,name}.
        Works for both backends.
        """
        if self.backend == "hub":
            # Pass a PIL image; hub model handles preprocessing
            pil = PILImage.fromarray(img_rgb)
            results = self.model(pil, size=self.input_size)
            dets = []
            if hasattr(results, "xyxy") and len(results.xyxy) > 0:
                preds = results.xyxy[0].cpu().numpy()
                for x1, y1, x2, y2, conf, cls in preds:
                    if float(conf) < self.conf_thres:
                        continue
                    cls = int(cls)
                    name = self.names[cls] if cls in self.names else str(cls)
                    dets.append({
                        "x1": int(round(x1)), "y1": int(round(y1)),
                        "x2": int(round(x2)), "y2": int(round(y2)),
                        "conf": float(conf), "cls": cls, "name": name
                    })
            return dets

        elif self.backend == "raw":
            # Manual preprocessing + NMS just like the repo
            im = self.letterbox(img_rgb, new_shape=self.input_size, stride=32, auto=True)[0]
            im = im.transpose((2, 0, 1))  # HWC->CHW
            im = np.ascontiguousarray(im)
            im = self.torch.from_numpy(im).to(self.device).float() / 255.0
            if im.ndim == 3:
                im = im.unsqueeze(0)

            pred = self.model(im)  # raw forward
            pred = self.nms(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]

            out = []
            if pred is None or len(pred) == 0:
                return out

            pred[:, :4] = self.scale_boxes(im.shape[2:], pred[:, :4], img_rgb.shape).round()
            for *xyxy, conf, cls in pred.cpu().tolist():
                cls = int(cls)
                name = self.names[cls] if cls in self.names else str(cls)
                x1, y1, x2, y2 = map(int, xyxy)
                out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "conf": float(conf), "cls": cls, "name": name})
            return out

        else:
            raise RuntimeError("YOLO backend not initialized")


class StereoConeDetector(Node):
    def __init__(self):
        super().__init__('stereo_cone_detector')

        # Parameters (adjust topics/paths as needed)
        self.declare_parameter('left_image_topic', '/zed/left/image_rect_color')
        self.declare_parameter('right_image_topic', '/zed/right/image_rect_color')
        self.declare_parameter('image_transport', 'raw')  # or 'compressed'
        self.declare_parameter('baseline', 0.12)
        self.declare_parameter('focal_length_px', 700)
        self.declare_parameter('yolo_repo_rel', 'yolov5')
        self.declare_parameter('yolo_weights_rel', 'yolov5/weights/best.pt')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('orb_n_features', 2500)
        self.declare_parameter('lowe_ratio', 0.8)

        self.bridge = CvBridge()
        base_dir = Path(__file__).parent.resolve()
        repo_dir = base_dir / self.get_parameter('yolo_repo_rel').get_parameter_value().string_value
        weights_path = base_dir / self.get_parameter('yolo_weights_rel').get_parameter_value().string_value
        device = self.get_parameter('device').get_parameter_value().string_value

        self.baseline = self.get_parameter('baseline').get_parameter_value().double_value
        self.focal_length = self.get_parameter('focal_length_px').get_parameter_value().integer_value
        self.ORB_N_FEATURES = self.get_parameter('orb_n_features').get_parameter_value().integer_value
        self.LOWE_RATIO = self.get_parameter('lowe_ratio').get_parameter_value().double_value

        self.yolo = YOLOv5Detector(repo_dir=repo_dir, weights_path=weights_path, device=device)

        # Setup ORB
        self.orb = cv2.ORB_create(self.ORB_N_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Topics
        left_topic = self.get_parameter('left_image_topic').get_parameter_value().string_value
        right_topic = self.get_parameter('right_image_topic').get_parameter_value().string_value
        img_transport = self.get_parameter('image_transport').get_parameter_value().string_value

        # QoS: BEST_EFFORT for sensor streams
        sensor_qos = qos_profile_sensor_data  # (BEST_EFFORT, KEEP_LAST, small depth)

        # Subscribers
        if img_transport == 'compressed':
            self.left_sub = self.create_subscription(
                RosCompressedImage, left_topic, self.left_cb_compressed, sensor_qos
            )
            self.right_sub = self.create_subscription(
                RosCompressedImage, right_topic, self.right_cb_compressed, sensor_qos
            )
            self.get_logger().info('Subscribed to compressed stereo images (BEST_EFFORT QoS)')
        else:
            self.left_sub = self.create_subscription(
                RosImage, left_topic, self.left_cb_raw, sensor_qos
            )
            self.right_sub = self.create_subscription(
                RosImage, right_topic, self.right_cb_raw, sensor_qos
            )
            self.get_logger().info('Subscribed to raw stereo images (BEST_EFFORT QoS)')

        # Publisher for cone pairs (visualization markers here) – keep RELIABLE
        self.marker_pub = self.create_publisher(MarkerArray, '/cone_pairs', 10)

        self.left_img = None
        self.right_img = None

    def left_cb_raw(self, msg):
        self.left_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_if_ready()

    def right_cb_raw(self, msg):
        self.right_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_if_ready()

    def left_cb_compressed(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.left_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.process_if_ready()

    def right_cb_compressed(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.right_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.process_if_ready()

    def process_if_ready(self):
        if self.left_img is None or self.right_img is None:
            return

        # Convert to RGB
        left_rgb = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(self.right_img, cv2.COLOR_BGR2RGB)

        # YOLO detection
        dets_left = self.yolo.infer(left_rgb)
        dets_right = self.yolo.infer(right_rgb)

        boxes_left = [(d['cls'], d['x1'], d['y1'], d['x2'], d['y2']) for d in dets_left]
        boxes_right = [(d['cls'], d['x1'], d['y1'], d['x2'], d['y2']) for d in dets_right]

        # ORB computation
        gray_left = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(self.right_img, cv2.COLOR_BGR2GRAY)
        kp_left, des_left = self.orb.detectAndCompute(gray_left, None)
        kp_right, des_right = self.orb.detectAndCompute(gray_right, None)

        if des_left is None or des_right is None:
            self.get_logger().warn('No ORB descriptors found in one of images')
            return

        matches = self.bf.knnMatch(des_left, des_right, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.LOWE_RATIO * n.distance:
                good_matches.append(m)

        # Group matches by box pairs
        def inside(x, y, box):
            x1, y1, x2, y2 = box
            return x1 <= x <= x2 and y1 <= y <= y2

        matches_by_pair = {}

        for match in good_matches:
            xL, yL = map(int, kp_left[match.queryIdx].pt)
            xR, yR = map(int, kp_right[match.trainIdx].pt)
            for clsL, x1L, y1L, x2L, y2L in boxes_left:
                if inside(xL, yL, (x1L, y1L, x2L, y2L)):
                    for clsR, x1R, y1R, x2R, y2R in boxes_right:
                        if clsL == clsR and inside(xR, yR, (x1R, y1R, x2R, y2R)):
                            key = ((x1L, y1L, x2L, y2L), (x1R, y1R, x2R, y2R), clsL)
                            matches_by_pair.setdefault(key, []).append(((xL, yL), (xR, yR)))
                            break
                    break

        # Draw and publish markers
        combined = np.hstack((self.left_img.copy(), self.right_img.copy()))
        offset_x = self.left_img.shape[1]
        markers = MarkerArray()
        marker_id = 0

        for pair_id, (key, points) in enumerate(matches_by_pair.items()):
            boxL, boxR, cls = key
            color = tuple(np.random.randint(0, 255, 3).tolist())
            color_norm = [c / 255.0 for c in color]

            # Draw bounding boxes and matches
            cv2.rectangle(combined, (boxL[0], boxL[1]), (boxL[2], boxL[3]), color, 2)
            cv2.rectangle(combined, (boxR[0] + offset_x, boxR[1]), (boxR[2] + offset_x, boxR[3]), color, 2)

            disparities = []
            for (ptL, ptR) in points:
                xL, yL = ptL
                xR, yR = ptR
                disparity = abs(xL - xR)
                if disparity > 0:
                    disparities.append(disparity)
                cv2.circle(combined, (xL, yL), 3, color, -1)
                cv2.circle(combined, (xR + offset_x, yR), 3, color, -1)
                cv2.line(combined, (xL, yL), (xR + offset_x, yR), color, 1)

            if disparities:
                avg_disp = float(np.mean(disparities))
                distance_m = (self.focal_length * self.baseline) / avg_disp
                dist_text = f"{distance_m:.2f} m"

                # Publish marker for cone position (assuming y=0, x=distance_m forward)
                marker = Marker()
                marker.header.frame_id = "camera_link"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(distance_m)
                marker.pose.position.y = 0.0
                marker.pose.position.z = 0.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                marker.color.r = color_norm[0]
                marker.color.g = color_norm[1]
                marker.color.b = color_norm[2]
                markers.markers.append(marker)
            else:
                dist_text = "No match"

            mid_x = (boxL[0] + boxL[2]) // 2
            label_pos = (mid_x, max(20, boxL[1] - 10))
            cv2.putText(
                combined, f"Pair {pair_id + 1}: {dist_text}", label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # Publish all cone markers
        self.marker_pub.publish(markers)

        # Show combined annotated stereo image
        cv2.imshow("Stereo Cone Detection", combined)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = StereoConeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
