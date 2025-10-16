#!/usr/bin/env python3
import os, io, math, time, json, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import cv2
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CompressedImage as RosCompressedImage

# ─────────────────────────────────────────────────────────────────────────────
#                       Small helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def yaw_from_quat(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def dot(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return a[0]*b[0] + a[1]*b[1]

def wrap_to_2pi(a: float) -> float:
    return a % (2.0*math.pi)

def unit_vec_from_yaw(yaw: float) -> Tuple[float,float]:
    return (math.cos(yaw), math.sin(yaw))

def right_normal_from_yaw(yaw: float) -> Tuple[float,float]:
    return (math.sin(yaw), -math.cos(yaw))

def left_normal_from_yaw(yaw: float) -> Tuple[float,float]:
    return (-math.sin(yaw), math.cos(yaw))


# ─────────────────────────────────────────────────────────────────────────────
#                              YOLOv5 wrapper
#   - Looks for repo at ../yolov5 and weights at ../yolov5/weights/best.pt
#   - Works on RGB np.uint8
# ----------------------------------------------------------------------------- 
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

            self.model = self.DetectMultiBackend(str(self.weights_path), device=self.device, dnn=False, data=None, fp16=False)
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

    @staticmethod
    def draw_rgb(img_rgb: np.ndarray, dets):
        for d in dets:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            label = f'{d["name"]} {d["conf"]:.2f}'
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0,255,0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_rgb, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0,255,0), -1)
            cv2.putText(img_rgb, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        return img_rgb

    @staticmethod
    def dets_to_yolo_txt(dets, img_w: int, img_h: int):
        lines = []
        for d in dets:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            cls = int(d["cls"])
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w  = (x2 - x1) / float(img_w)
            h  = (y2 - y1) / float(img_h)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines


# ─────────────────────────────────────────────────────────────────────────────
#                      Step-and-shoot Fixed Route Driver
#   Differences vs your reference:
#   - save dir -> dataset_bbox
#   - add YOLO; repo path '../yolov5'
#   - distance-gated capture: step_m (default 0.1 m)
#   - when within slop with L/R frames: stop, run YOLO, save annotated + labels
# ─────────────────────────────────────────────────────────────────────────────
class FixedRouteStepper(Node):
    def __init__(self):
        super().__init__("fixed_route_yolo_stepper", automatically_declare_parameters_from_overrides=True)

        # -------- Params (all your originals + stepper/YOLO bits) --------
        self.declare_parameter("wheelbase_m", 1.5)
        self.declare_parameter("radius_m", 9.125)
        self.declare_parameter("target_speed_mps", 3.0)      # slower is fine for stepper
        self.declare_parameter("steering_right_sign", -1.0)
        self.declare_parameter("control_hz", 50.0)

        self.declare_parameter("accel_gain", 1.5)
        self.declare_parameter("accel_limit", 2.0)

        self.declare_parameter("straight1_m", 16.0)
        self.declare_parameter("straight2_m", 25.0)

        self.declare_parameter("odom_topic", "/ground_truth/odom")
        self.declare_parameter("cmd_topic", "/cmd")

        self.declare_parameter("left_image_topic", "/zed/left/image_rect_color")
        self.declare_parameter("right_image_topic", "/zed/right/image_rect_color")
        self.declare_parameter("image_transport", "raw")     # "raw" or "compressed"

        # stepper & pairing
        self.declare_parameter("step_m", 0.10)               # MOVE 0.1m, stop, shoot, repeat
        self.declare_parameter("stop_speed_eps", 0.05)       # m/s threshold to consider stopped
        self.declare_parameter("sync_slop_ms", 120.0)        # L/R time tolerance
        self.declare_parameter("rb_max", 50)
        self.declare_parameter("qos_best_effort", True)

        # YOLO paths (relative to parent dir)
        self.declare_parameter("yolo_repo_rel", "yolov5")
        self.declare_parameter("yolo_weights_rel", "yolov5/weights/best.pt")
        self.declare_parameter("yolo_device", "cuda")

        gp = self.get_parameter
        f = lambda k: float(getattr(gp(k).get_parameter_value(), "double_value",
                                    gp(k).get_parameter_value().integer_value))
        i = lambda k: int(getattr(gp(k).get_parameter_value(), "integer_value",
                                   gp(k).get_parameter_value().double_value))
        s = lambda k: gp(k).get_parameter_value().string_value
        b = lambda k: gp(k).get_parameter_value().bool_value

        self.wheelbase       = f("wheelbase_m")
        self.radius          = f("radius_m")
        self.v_target_move   = f("target_speed_mps")
        self.right_sign      = f("steering_right_sign")
        self.control_hz      = f("control_hz")
        self.accel_gain      = f("accel_gain")
        self.accel_limit     = f("accel_limit")

        self.seg1_len        = f("straight1_m")
        self.seg2_len        = f("straight2_m")
        self.odom_topic      = s("odom_topic")
        self.cmd_topic       = s("cmd_topic")

        self.left_topic      = s("left_image_topic")
        self.right_topic     = s("right_image_topic")
        self.image_transport = s("image_transport").lower()

        self.step_m          = f("step_m")
        self.stop_speed_eps  = f("stop_speed_eps")
        self.sync_slop_ms    = f("sync_slop_ms")
        self.rb_max          = i("rb_max")
        self.qos_best_effort = b("qos_best_effort")

        # Paths
        base = Path(__file__).parent
        self.save_root   = (base / "dataset_bbox"); self.save_root.mkdir(parents=True, exist_ok=True)
        self.left_dir    = (self.save_root / "left_cam");  self.left_dir.mkdir(parents=True, exist_ok=True)
        self.right_dir   = (self.save_root / "right_cam"); self.right_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir  = (self.save_root / "labels");    self.labels_dir.mkdir(parents=True, exist_ok=True)

        # YOLO init
        repo_dir    = (base / s("yolo_repo_rel")).resolve()
        weights_pt  = (base / s("yolo_weights_rel")).resolve()
        self.get_logger().info(f"[yolo] repo={repo_dir}  weights={weights_pt}")
        self.yolo = YOLOv5Detector(repo_dir=repo_dir, weights_path=weights_pt, device=s("yolo_device"))

        # -------- Live image buffers (RGB frames + stamps) --------
        # store numpy RGB images to run YOLO
        self.left_buf:  List[Tuple[int, np.ndarray]] = []    # (stamp_ns, rgb)
        self.right_buf: List[Tuple[int, np.ndarray]] = []

        # -------- State machine --------
        self.state = "WAIT_INIT"      # route state
        self.substate = "MOVE"        # MOVE or CAPTURE within route state

        self.p0 = (0.0, 0.0); self.yaw0 = 0.0; self.fwd0 = (1.0, 0.0); self.s1_progress = 0.0
        self.p1 = (0.0, 0.0); self.yaw1 = 0.0; self.center = (0.0, 0.0); self.theta_start = 0.0; self.theta_progress = 0.0
        self.p2 = (0.0, 0.0); self.yaw2 = 0.0; self.fwd2 = (1.0, 0.0); self.s2_progress = 0.0

        # step checkpoints within current segment
        self.next_step_at = 0.0   # meters along current segment where next stop+capture happens

        # odom
        self.have_odom = False
        self.pos = (0.0, 0.0); self.yaw = 0.0; self.speed = 0.0

        # fixed circle steer for CIRCLE segment
        self.delta_circle = math.atan(self.wheelbase / max(0.01, self.radius)) * self.right_sign

        # -------- I/O --------
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if self.qos_best_effort else QoSReliabilityPolicy.RELIABLE
        )
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos)
        self.pub_ack = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)

        if self.image_transport == "compressed":
            self.create_subscription(RosCompressedImage, self.left_topic,  self._left_compressed_cb,  qos)
            self.create_subscription(RosCompressedImage, self.right_topic, self._right_compressed_cb, qos)
            self.get_logger().info(f"[img] subs (Compressed): {self.left_topic} + {self.right_topic}")
        else:
            self.create_subscription(RosImage, self.left_topic,  self._left_raw_cb,  qos)
            self.create_subscription(RosImage, self.right_topic, self._right_raw_cb, qos)
            self.get_logger().info(f"[img] subs (Raw): {self.left_topic} + {self.right_topic}")

        self.get_logger().info(
            f"[route] wheelbase={self.wheelbase:.3f} m, R={self.radius:.3f} m, delta_circle={self.delta_circle:.3f} rad"
        )

        self.timer = self.create_timer(1.0 / max(1.0, self.control_hz), self._tick)

        # debug counters
        self._pairs_saved = 0
        self._last_log_t = time.time()

    # ─────────── Callbacks ───────────
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position; q = msg.pose.pose.orientation
        self.pos = (float(p.x), float(p.y))
        self.yaw = yaw_from_quat(q.w, q.x, q.y, q.z)
        vx = float(msg.twist.twist.linear.x); vy = float(msg.twist.twist.linear.y)
        self.speed = math.hypot(vx, vy)
        self.have_odom = True

    # store RGB frames into buffers
    def _left_raw_cb(self, msg: RosImage):
        stamp = self._stamp_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
        rgb = self._rgb_from_raw(msg)
        if rgb is not None:
            self.left_buf.append((stamp, rgb))
            if len(self.left_buf) > self.rb_max: del self.left_buf[:len(self.left_buf)-self.rb_max]

    def _right_raw_cb(self, msg: RosImage):
        stamp = self._stamp_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
        rgb = self._rgb_from_raw(msg)
        if rgb is not None:
            self.right_buf.append((stamp, rgb))
            if len(self.right_buf) > self.rb_max: del self.right_buf[:len(self.right_buf)-self.rb_max]

    def _left_compressed_cb(self, msg: RosCompressedImage):
        stamp = self._stamp_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
        rgb = self._rgb_from_compressed(msg)
        if rgb is not None:
            self.left_buf.append((stamp, rgb))
            if len(self.left_buf) > self.rb_max: del self.left_buf[:len(self.left_buf)-self.rb_max]

    def _right_compressed_cb(self, msg: RosCompressedImage):
        stamp = self._stamp_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
        rgb = self._rgb_from_compressed(msg)
        if rgb is not None:
            self.right_buf.append((stamp, rgb))
            if len(self.right_buf) > self.rb_max: del self.right_buf[:len(self.right_buf)-self.rb_max]

    # ─────────── Control & stepper ───────────
    def _tick(self):
        if not self.have_odom:
            # gentle nudge to get moving when odom not ready
            self._publish_cmd(accel=min(self.accel_limit, 0.5*self.accel_limit), steering=0.0)
            return

        steering_cmd = 0.0
        v_target = 0.0  # default to stopped unless in MOVE

        # Route state transitions
        if self.state == "WAIT_INIT":
            self.p0 = self.pos; self.yaw0 = self.yaw; self.fwd0 = unit_vec_from_yaw(self.yaw0)
            self.s1_progress = 0.0
            self.next_step_at = 0.0
            self.state = "STRAIGHT1"; self.substate = "MOVE"
            self.get_logger().info("[state] STRAIGHT1 start")

        elif self.state == "STRAIGHT1":
            d = (self.pos[0] - self.p0[0], self.pos[1] - self.p0[1])
            self.s1_progress = max(0.0, dot(d, self.fwd0))
            steering_cmd = 0.0
            if self.s1_progress >= self.seg1_len:
                # enter circle
                self.p1 = self.pos; self.yaw1 = self.yaw
                n_hat = right_normal_from_yaw(self.yaw1) if self.right_sign < 0 else left_normal_from_yaw(self.yaw1)
                self.center = (self.p1[0] + self.radius * n_hat[0], self.p1[1] + self.radius * n_hat[1])
                self.theta_start = math.atan2(self.p1[1] - self.center[1], self.p1[0] - self.center[0])
                self.theta_progress = 0.0
                self.next_step_at = 0.0
                self.state = "CIRCLE"; self.substate = "MOVE"
                self.get_logger().info(f"[state] CIRCLE start center={self.center} theta0={self.theta_start:.3f}")

        elif self.state == "CIRCLE":
            steering_cmd = self.delta_circle
            theta_now = math.atan2(self.pos[1] - self.center[1], self.pos[0] - self.center[0])
            if self.right_sign < 0:
                self.theta_progress = wrap_to_2pi(self.theta_start - theta_now)
            else:
                self.theta_progress = wrap_to_2pi(theta_now - self.theta_start)

            if self.theta_progress >= (2.0 * math.pi - 1e-2):
                # exit circle to straight2
                self.p2 = self.pos; self.yaw2 = self.yaw; self.fwd2 = unit_vec_from_yaw(self.yaw2)
                self.s2_progress = 0.0
                self.next_step_at = 0.0
                self.state = "STRAIGHT2"; self.substate = "MOVE"
                self.get_logger().info("[state] STRAIGHT2 start")

        elif self.state == "STRAIGHT2":
            d = (self.pos[0] - self.p2[0], self.pos[1] - self.p2[1])
            self.s2_progress = max(0.0, dot(d, self.fwd2))
            steering_cmd = 0.0
            if self.s2_progress >= self.seg2_len:
                self.state = "DONE"; self.substate = "STOPPED"
                self.get_logger().info("[state] DONE; braking to stop")

        else:  # DONE
            v_target = 0.0
            self._track_speed(v_target=v_target, steering=0.0)
            return

        # ── Step-and-shoot substate ──
        seg_progress, seg_len = self._current_segment_progress_len()
        if self.substate == "MOVE":
            # keep moving until next_step_at
            v_target = self.v_target_move
            if seg_progress >= self.next_step_at - 1e-6:
                # reached step point → stop to capture
                self.substate = "CAPTURE"
                self.get_logger().info(f"[capture] Reached step {self.next_step_at:.2f} m → stopping")
        elif self.substate == "CAPTURE":
            v_target = 0.0
            if self.speed <= self.stop_speed_eps:
                # stopped: try take a synced pair
                ok = self._capture_and_save_pair()
                if ok:
                    self._pairs_saved += 1
                    self.next_step_at += self.step_m
                    # if next step beyond segment len, we still move; route state will switch soon
                    self.substate = "MOVE"
                    self.get_logger().info(f"[capture] saved={self._pairs_saved} next_step_at={self.next_step_at:.2f} m")
                else:
                    # couldn't pair within slop yet — keep waiting (still stopped)
                    pass

        # Push accel to target speed
        self._track_speed(v_target=v_target, steering=steering_cmd)

        # periodic debug
        now = time.time()
        if now - self._last_log_t >= 2.0:
            self.get_logger().info(
                f"[dbg] state={self.state}/{self.substate} speed={self.speed:.2f} "
                f"progress={seg_progress:.2f}/{seg_len:.2f}m  pairs={self._pairs_saved} "
                f"bufL={len(self.left_buf)} bufR={len(self.right_buf)}"
            )
            self._last_log_t = now

    def _track_speed(self, v_target: float, steering: float):
        v_err = v_target - self.speed
        accel_cmd = max(-self.accel_limit, min(self.accel_limit, self.accel_gain * v_err))
        self._publish_cmd(accel=accel_cmd, steering=steering)

    def _current_segment_progress_len(self) -> Tuple[float, float]:
        if self.state == "STRAIGHT1":
            return (self.s1_progress, self.seg1_len)
        if self.state == "CIRCLE":
            return (self.theta_progress * self.radius, 2.0 * math.pi * self.radius)
        if self.state == "STRAIGHT2":
            return (self.s2_progress, self.seg2_len)
        # WAIT_INIT / DONE
        return (0.0, 1.0)

    # ─────────── Capture pipeline ───────────
    def _capture_and_save_pair(self) -> bool:
        """Find nearest-time L/R frames within slop, run YOLO, save images + labels. Return True on success."""
        if not self.left_buf or not self.right_buf:
            return False
        # use the most recent timestamp as anchor
        l_stamp = self.left_buf[-1][0]
        idx_r, diff_r = self._nearest_index(self.right_buf, l_stamp)
        if idx_r is None or diff_r > self._slop_ns():
            # try reverse (anchor on right)
            r_stamp = self.right_buf[-1][0]
            idx_l, diff_l = self._nearest_index(self.left_buf, r_stamp)
            if idx_l is None or diff_l > self._slop_ns():
                return False
            # got L near R
            stamp = r_stamp
            rgb_l = self.left_buf[idx_l][1]
            rgb_r = self.right_buf[-1][1]
        else:
            # got R near L
            stamp = l_stamp
            rgb_l = self.left_buf[-1][1]
            rgb_r = self.right_buf[idx_r][1]

        # Run YOLO
        dets_l = self.yolo.infer(rgb_l)
        dets_r = self.yolo.infer(rgb_r)

        # Draw & save annotated images
        ann_l = self.yolo.draw_rgb(rgb_l.copy(), dets_l)
        ann_r = self.yolo.draw_rgb(rgb_r.copy(), dets_r)

        l_path = self.left_dir  / f"L_{stamp}.jpg"
        r_path = self.right_dir / f"R_{stamp}.jpg"
        # write as JPEG (convert RGB->BGR for cv2 imwrite)
        cv2.imwrite(str(l_path), cv2.cvtColor(ann_l, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(r_path), cv2.cvtColor(ann_r, cv2.COLOR_RGB2BGR))

        # Save YOLO label .txt (normalized)
        lh, lw = ann_l.shape[0], ann_l.shape[1]
        rh, rw = ann_r.shape[0], ann_r.shape[1]
        l_txt = self.labels_dir / f"L_{stamp}.txt"
        r_txt = self.labels_dir / f"R_{stamp}.txt"
        l_lines = self.yolo.dets_to_yolo_txt(dets_l, lw, lh)
        r_lines = self.yolo.dets_to_yolo_txt(dets_r, rw, rh)
        l_txt.write_text("\n".join(l_lines) + ("\n" if l_lines else ""), encoding="utf-8")
        r_txt.write_text("\n".join(r_lines) + ("\n" if r_lines else ""), encoding="utf-8")

        self.get_logger().info(f"[img] saved pair #{self._pairs_saved+1} -> base={stamp}  "
                               f"left={l_path.name} right={r_path.name} labels=2")
        return True

    # ─────────── Publish helper ───────────
    def _publish_cmd(self, accel: float, steering: float):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        msg.drive.acceleration   = float(accel)
        msg.drive.speed          = 0.0
        self.pub_ack.publish(msg)

    # ─────────── Utils ───────────
    def _nearest_index(self, buf: List[Tuple[int, np.ndarray]], target: int):
        if not buf: return (None, None)
        best_i, best_d = None, 1<<62
        for i in range(len(buf)-1, -1, -1):
            d = abs(buf[i][0] - target)
            if d < best_d:
                best_d = d; best_i = i
            if buf[i][0] < target and best_i is not None and d > best_d:
                break
        return (best_i, best_d)

    def _slop_ns(self) -> int:
        return int(self.sync_slop_ms * 1e6)

    @staticmethod
    def _stamp_ns(sec: int, nsec: int) -> int:
        return int(sec) * 10**9 + int(nsec)

    def _rgb_from_raw(self, msg: RosImage) -> Optional[np.ndarray]:
        h, w = int(msg.height), int(msg.width)
        enc = (msg.encoding or "rgb8").lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if enc in ("rgb8", "bgr8", "rgba8", "bgra8"):
            channels = 4 if "a8" in enc else 3
            expected = w * channels
            if msg.step == expected:
                arr = data.reshape(h, w, channels)
            else:
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, msg.step)[:, :expected].reshape(h, w, channels)
            if enc.startswith("bgr") or enc.startswith("bgra"):
                arr = arr[..., :3][:, :, ::-1]   # BGR(A) -> RGB
            elif enc.endswith("a8"):
                arr = arr[..., :3]
            return arr if arr.dtype == np.uint8 else arr.astype(np.uint8)
        elif enc in ("mono8", "8uc1"):
            if msg.step == w:
                arr = data.reshape(h, w)
            else:
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, msg.step)[:, :w]
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        else:
            expected = w * 3
            arr = data.reshape(h, msg.step)[:, :expected].reshape(h, w, 3)
            return arr

    def _rgb_from_compressed(self, msg: RosCompressedImage) -> Optional[np.ndarray]:
        bio_in = io.BytesIO(bytes(msg.data))
        pil = PILImage.open(bio_in).convert("RGB")
        return np.array(pil, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#                                       main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = FixedRouteStepper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
