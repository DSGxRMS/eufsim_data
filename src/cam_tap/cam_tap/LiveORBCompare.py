#!/usr/bin/env python3
"""
LiveORB_Continuous_Output.py

ROS2 node modified to continuously calculate cone 3D coordinates 
and print the results array to the terminal using a ROS Timer.
"""
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import math
from pathlib import Path
from typing import Tuple, List
import time # Time needed for printing timestamp

# --- Imports for ROS Message Types ---
from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import Odometry
# --------------------------------------------------------------------

# ----------------- Parameters (kept the same) -----------------
LEFT_IMAGE_TOPIC = "/zed/left/image_rect_color"
RIGHT_IMAGE_TOPIC = "/zed/right/image_rect_color"
ODOM_TOPIC = "/ground_truth/odom"
YOLO_REPO_REL = "yolov5"
YOLO_WEIGHTS_REL = "yolov5/weights/best.pt" 

FX = 448.13386274345095
FY = 448.13386274345095
CX = 640.5
CY = 360.5
BASELINE = 0.06

ORB_N_FEATURES = 2500
LOWE_RATIO = 0.8
MIN_MATCHES_PER_BOX = 3

# Console output frequency (Hz)
OUTPUT_HZ = 5.0 

# Color map for classes (only used for output text)
CLASS_NAMES = {
    0: "Blue", 1: "Yellow", 2: "Orange", 
    3: "Big Orange", 4: "Unknown", -1: "Proximity Group"
}

# ----------------- Helper Functions (YOLO, xyz_from_cone, etc. - kept the same) -----------------
# (Skipping helper class/functions here for brevity, assume they are included)

def import_cones_msg():
    # Helper kept for completeness, though GT is unused for output
    try:
        from eufs_msgs.msg import ConeArrayWithCovariance as ConesMsg
        return ConesMsg
    except Exception:
        try:
            from eufs_msgs.msg import ConeArray as ConesMsg
            return ConesMsg
        except Exception:
            return None

def xyz_from_cone(cone) -> Tuple[float, float, float]:
    for attr in ("point", "position", "location"):
        if hasattr(cone, attr):
            p = getattr(cone, attr)
            return float(getattr(p, "x", 0.0)), float(getattr(p, "y", 0.0)), float(getattr(p, "z", 0.0))
    if hasattr(cone, "x") and hasattr(cone, "y"):
        return float(cone.x), float(cone.y), float(getattr(cone, "z", 0.0))
    return 0.0, 0.0, 0.0

# ----------------- YOLO loader (Assumed Present and Correct) -----------------
class YOLOv5Detector:
    # ... (YOLOv5Detector class code remains the same as previous versions) ...
    def __init__(self, repo_dir: Path, weights_path: Path, device: str = "cpu"):
        import torch
        self.repo_dir = Path(repo_dir).resolve()
        self.weights_path = Path(weights_path).resolve()
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.backend = None
        self.input_size = 640
        self.conf_thres = 0.35
        self.iou_thres = 0.45
        self.names = {}
        # Try loading via torch.hub (local)
        try:
            self.model = torch.hub.load(str(self.repo_dir), 'custom', path=str(self.weights_path), source='local').to(self.device)
            self.model.eval()
            self.names = self.model.names if hasattr(self.model, "names") else {}
            self.backend = "hub"
            print("[YOLO] Loaded via torch.hub (local).")
            return
        except Exception as e:
            pass
        # Try raw in-repo loading (DetectMultiBackend)
        try:
            import sys
            sys.path.insert(0, str(self.repo_dir))
            from models.common import DetectMultiBackend
            from utils.augmentations import letterbox
            from utils.general import non_max_suppression, scale_coords
            self.DetectMultiBackend = DetectMultiBackend
            self.letterbox = letterbox
            self.nms = non_max_suppression
            self.scale_coords = scale_coords
            self.model = DetectMultiBackend(str(self.weights_path), device=self.device)
            self.names = getattr(self.model, "names", {})
            self.backend = "raw"
            print("[YOLO] Loaded via raw DetectMultiBackend.")
            return
        except Exception as e:
            raise RuntimeError("Failed to load YOLOv5 model. Check repo & weights paths.")

    def infer(self, img_rgb: np.ndarray):
        if img_rgb is None: return []
        if self.backend == "hub":
            results = self.model(img_rgb, size=self.input_size)
            dets = []
            if hasattr(results, "xyxy") and len(results.xyxy):
                preds = results.xyxy[0].cpu().numpy()
                for x1, y1, x2, y2, conf, cls in preds:
                    if float(conf) < self.conf_thres: continue
                    dets.append({"x1": int(round(x1)), "y1": int(round(y1)),
                                 "x2": int(round(x2)), "y2": int(round(y2)),
                                 "conf": float(conf), "cls": int(cls)})
            return dets
        elif self.backend == "raw":
            im = self.letterbox(img_rgb, new_shape=self.input_size)[0]
            im = im[:, :, ::-1].transpose(2, 0, 1) if False else im.transpose(2,0,1)
            im = np.ascontiguousarray(im)
            import torch
            im_t = torch.from_numpy(im).to(self.device).float() / 255.0
            if im_t.ndim == 3: im_t = im_t.unsqueeze(0)
            pred = self.model(im_t)
            pred = self.nms(pred, self.conf_thres, self.iou_thres)[0]
            out = []
            if pred is None: return out
            pred[:, :4] = self.scale_coords(im_t.shape[2:], pred[:, :4], img_rgb.shape).round()
            for *xyxy, conf, cls in pred.cpu().tolist():
                x1, y1, x2, y2 = map(int, xyxy)
                out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(conf), "cls": int(cls)})
            return out
        else:
            return []


# ----------------- Main Node (Continuous Output) -----------------
class LiveORBCompareNode(Node):
    def __init__(self):
        super().__init__("liveorb_continuous_output")
        self.get_logger().info("Starting LiveORB Continuous Output node")

        # --- Parameters and Setup ---
        self.declare_parameter("left_image_topic", LEFT_IMAGE_TOPIC)
        self.declare_parameter("right_image_topic", RIGHT_IMAGE_TOPIC)
        self.declare_parameter("odom_topic", ODOM_TOPIC)
        self.declare_parameter("yolo_repo", YOLO_REPO_REL)
        self.declare_parameter("yolo_weights", YOLO_WEIGHTS_REL)
        self.declare_parameter("baseline", BASELINE)
        self.declare_parameter("fx", FX)
        self.declare_parameter("fy", FY)
        self.declare_parameter("cx", CX)
        self.declare_parameter("cy", CY)

        self.left_topic = self.get_parameter("left_image_topic").value
        self.right_topic = self.get_parameter("right_image_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value

        base = Path(__file__).parent.resolve()
        repo_dir = base / self.get_parameter("yolo_repo").value
        weights_path = base / self.get_parameter("yolo_weights").value

        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)
        self.baseline = float(self.get_parameter("baseline").value)

        # YOLO, ORB, CV bridge setup
        try:
            self.yolo = YOLOv5Detector(repo_dir, weights_path, device="cpu")
        except Exception as e:
            self.get_logger().warning("YOLO failed to load: " + str(e))
            self.yolo = None

        self.orb = cv2.ORB_create(ORB_N_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.bridge = CvBridge()

        # --- Data Storage ---
        self.data_lock = threading.Lock() # Lock is necessary as callbacks run in parallel threads
        self.left_img = None
        self.right_img = None
        self.odom_pose = (0.0, 0.0, 0.0)

        # --- Subscribers setup (Uses imported class objects) ---
        qos = qos_profile_sensor_data
        
        self.left_sub = self.create_subscription(RosImage, self.left_topic, self.left_cb, qos)
        self.right_sub = self.create_subscription(RosImage, self.right_topic, self.right_cb, qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos)
        
        # --- Timer for Continuous Calculation and Output ---
        self.output_timer = self.create_timer(1.0 / OUTPUT_HZ, self.timer_output_cb)

    # ------------- Callbacks (Update Shared Data Safely) -------------
    
    def left_cb(self, msg: RosImage): 
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.data_lock:
                # Copy image data for thread safety
                self.left_img = cv_img.copy() 
        except Exception as e:
            self.get_logger().error("left_cb cv_bridge error: " + str(e))

    def right_cb(self, msg: RosImage):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.data_lock:
                self.right_img = cv_img.copy() 
        except Exception as e:
            self.get_logger().error("right_cb cv_bridge error: " + str(e))

    def odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        with self.data_lock:
            self.odom_pose = (float(px), float(py), float(yaw))

    # ----------------- Core Calculation Method (Runs on Timer) -----------------

    def calculate_cones(self, imgL, imgR, current_odom_pose) -> List[Tuple[float, float, float, int]]:
        """
        Runs stereo processing on collected data and returns a list of cones 
        (X_world, Y_world, Z_cam_depth, Class_ID).
        """
        if imgL is None or imgR is None or current_odom_pose == (0.0, 0.0, 0.0):
            return []

        # --- YOLO detections ---
        boxesL, boxesR = [], []
        if self.yolo is not None:
            try:
                detsL = self.yolo.infer(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
                detsR = self.yolo.infer(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
                boxesL = [(d['cls'], d['x1'], d['y1'], d['x2'], d['y2']) for d in detsL]
                boxesR = [(d['cls'], d['x1'], d['y1'], d['x2'], d['y2']) for d in detsR]
            except Exception as e:
                self.get_logger().warning("YOLO inference error: " + str(e))

        # ORB keypoints and matching
        grayL, grayR = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        kpL, desL = self.orb.detectAndCompute(grayL, None)
        kpR, desR = self.orb.detectAndCompute(grayR, None)
        if desL is None or desR is None: return []

        matches_knn = self.bf.knnMatch(desL, desR, k=2)
        
        # Structure good matches as (match_object, (xl, yl), (xr, yr))
        good_matches_data = []
        for pair in matches_knn:
            if len(pair) > 1 and pair[0].distance < LOWE_RATIO * pair[1].distance:
                m = pair[0]
                xl, yl = kpL[m.queryIdx].pt
                xr, yr = kpR[m.trainIdx].pt
                good_matches_data.append((m, (xl, yl), (xr, yr)))

        # Group matches by box/proximity
        matches_by_pair = {} 
        
        for m_obj, (xl, yl), (xr, yr) in good_matches_data:
            xl_i, yl_i, xr_i, yr_i = int(xl), int(yl), int(xr), int(yr)
            
            # --- Logic when YOLO boxes are available ---
            if boxesL and boxesR:
                for clsL, x1L, y1L, x2L, y2L in boxesL:
                    if x1L <= xl_i <= x2L and y1L <= yl_i <= y2L:
                        for clsR, x1R, y1R, x2R, y2R in boxesR:
                            if clsL == clsR and x1R <= xr_i <= x2R and y1R <= yr_i <= y2R:
                                key = ((x1L,y1L,x2L,y2L),(x1R,y1R,x2R,y2R),clsL)
                                matches_by_pair.setdefault(key, []).append(((int(round(xl)), int(round(yl))), (int(round(xr)), int(round(yr)))))
                                break
                        break
            
            # --- Logic when NO YOLO boxes are available (Proximity grouping) ---
            else:
                key = ((int(xl)-5,int(yl)-5,int(xl)+5,int(yl)+5),(int(xr)-5,int(yr)-5,int(xr)+5,int(yr)+5),-1)
                matches_by_pair.setdefault(key, []).append(((int(round(xl)), int(round(yl))), (int(round(xr)), int(round(yr)))))

        # Triangulation and World Transform
        final_cones_list = []
        ox, oy, oyaw = current_odom_pose
        for (_, _, cls), pts in matches_by_pair.items():
            if len(pts) < MIN_MATCHES_PER_BOX: continue
            
            left_us = np.array([p[0][0] for p in pts], dtype=float)
            left_vs = np.array([p[0][1] for p in pts], dtype=float)
            right_us = np.array([p[1][0] for p in pts], dtype=float)
            
            uL, vL, uR = float(left_us.mean()), float(left_vs.mean()), float(right_us.mean())
            disparity = uL - uR
            if disparity <= 0.1: continue
            
            Z_cam = (self.fx * self.baseline) / disparity
            X_cam = (uL - self.cx) * Z_cam / self.fx
            
            X_vehicle = float(Z_cam)
            Y_vehicle = float(-X_cam)
            
            Xw = ox + X_vehicle * math.cos(oyaw) - Y_vehicle * math.sin(oyaw)
            Yw = oy + X_vehicle * math.sin(oyaw) + Y_vehicle * math.cos(oyaw)
            
            final_cones_list.append((float(Xw), float(Yw), float(Z_cam), int(cls)))

        return final_cones_list

    # ----------------- Timer Callback (Output Logic) -----------------
    def timer_output_cb(self):
        # 1. Safely retrieve the latest data snapshot
        with self.data_lock:
            imgL_copy = self.left_img.copy() if self.left_img is not None else None
            imgR_copy = self.right_img.copy() if self.right_img is not None else None
            odom_pose_copy = self.odom_pose

        # 2. Run calculation
        cones_result = self.calculate_cones(imgL_copy, imgR_copy, odom_pose_copy)

        # 3. Print output
        current_time = time.strftime("%H:%M:%S")
        
        print("\n" + "=" * 60)
        print(f"[{current_time}] Calculated Cones ({len(cones_result)})")
        print("-" * 60)
        
        if cones_result:
            output_list = []
            for Xw, Yw, Z_cam, cls_id in cones_result:
                 cls_name = CLASS_NAMES.get(cls_id, "Unknown")
                 output_list.append(f"[{Xw:.2f}, {Yw:.2f}, {cls_id} ({cls_name})]")
            
            print(f"[{', '.join(output_list)}]")
        else:
            print("No cones detected or waiting for initial data...")
        print("=" * 60)

    def destroy_node(self):
        self.output_timer.cancel()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LiveORBCompareNode()
    
    # Run the ROS event loop continuously
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Fatal error during spin: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
