#!/usr/bin/env python3
"""
LiveORB_GT_Compare.py

ROS2 node that:
 - subscribes to left/right rectified images
 - runs YOLOv5 to detect cones (class/color)
 - runs ORB + FLANN matching to match left/right keypoints inside YOLO boxes
 - computes disparity -> full 3D (camera optical frame)
 - transforms optical -> vehicle frame -> odom/world using /ground_truth/odom
 - subscribes to /ground_truth/cones (ConeArrayWithCovariance | ConeArray)
 - displays Matplotlib BEV comparing GT (solid) vs LiveORB (hollow)
 - displays OpenCV stereo visualization

Notes:
 - Camera intrinsics are taken from data (fx=fy=448.13386274345095, cx=640.5, cy=360.5)
 - Baseline = 0.06 m (6 cm)
 - Camera optical frame is assumed standard ROS optical frame:
     X_cam -> right, Y_cam -> down, Z_cam -> forward
 - Mapping camera -> vehicle frame (x forward, y left, z up):
     X_vehicle = Z_cam
     Y_vehicle = -X_cam
     Z_vehicle = -Y_cam
"""
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math
import threading
import time

# ----------------- Parameters (tweak if needed) -----------------
LEFT_IMAGE_TOPIC = "/zed/left/image_rect_color"
RIGHT_IMAGE_TOPIC = "/zed/right/image_rect_color"
ODOM_TOPIC = "/ground_truth/odom"
GT_CONES_TOPIC = "/ground_truth/cones"

# YOLO paths (relative to this file's folder). Update if needed.
YOLO_REPO_REL = "yolov5"
YOLO_WEIGHTS_REL = "yolov5/weights/best.pt"  # change if your weights path differs

# Camera intrinsics (from your camera_info)
FX = 448.13386274345095
FY = 448.13386274345095
CX = 640.5
CY = 360.5

BASELINE = 0.06  # meters (6 cm)
CAMERA_X = 0.0   # camera position in vehicle frame (forward)
CAMERA_Y = 0.0   # camera lateral offset (left positive)
CAMERA_Z = 0.664 # camera height (meters) above vehicle origin (unused for BEV)

# ORB / matching params
ORB_N_FEATURES = 2500
LOWE_RATIO = 0.8
MIN_MATCHES_PER_BOX = 3

# BEV plotting params
BEV_XLIM = (-5, 40)   # forward
BEV_YLIM = (-15, 15)  # left/right
BEV_HZ = 10.0

# Color map for classes (BGR for OpenCV, hex for matplotlib)
CLASS_COLORS_BGR = {
    0: (255, 0, 0),       # blue (class 0)
    1: (0, 255, 255),     # yellow (class 1)
    2: (0, 165, 255),     # orange (class 2)
    3: (0, 69, 255),      # big_orange (class 3)
    4: (200, 200, 200)    # unknown (class 4)
}
CLASS_COLORS_MPL = {
    0: "tab:blue",
    1: "gold",
    2: "orange",
    3: "darkorange",
    4: "0.6"
}

# ----------------- Helper: import GT message robustly -----------------
def import_cones_msg():
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
    # Try common attributes
    for attr in ("point", "position", "location"):
        if hasattr(cone, attr):
            p = getattr(cone, attr)
            return float(getattr(p, "x", 0.0)), float(getattr(p, "y", 0.0)), float(getattr(p, "z", 0.0))
    if hasattr(cone, "x") and hasattr(cone, "y"):
        return float(cone.x), float(cone.y), float(getattr(cone, "z", 0.0))
    return 0.0, 0.0, 0.0

# ----------------- YOLO loader (supports yolov5 local repo) -----------------
class YOLOv5Detector:
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

        # Try torch.hub local load (yolov5)
        try:
            self.model = torch.hub.load(str(self.repo_dir), 'custom', path=str(self.weights_path), source='local').to(self.device)
            self.model.eval()
            self.names = self.model.names if hasattr(self.model, "names") else {}
            self.backend = "hub"
            print("[YOLO] Loaded via torch.hub (local).")
            return
        except Exception as e:
            print("[YOLO] torch.hub load failed:", e)

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
            print("[YOLO] raw backend load failed:", e)
            raise RuntimeError("Failed to load YOLOv5 model. Check repo & weights paths.")

    def infer(self, img_rgb: np.ndarray):
        """
        Return list of detections: dict(x1,y1,x2,y2,conf,cls)
        Works with both hub and raw backends (best-effort).
        """
        if img_rgb is None:
            return []
        if self.backend == "hub":
            results = self.model(img_rgb, size=self.input_size)
            dets = []
            if hasattr(results, "xyxy") and len(results.xyxy):
                preds = results.xyxy[0].cpu().numpy()
                for x1, y1, x2, y2, conf, cls in preds:
                    if float(conf) < self.conf_thres:
                        continue
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
            if im_t.ndim == 3:
                im_t = im_t.unsqueeze(0)
            pred = self.model(im_t)
            pred = self.nms(pred, self.conf_thres, self.iou_thres)[0]
            out = []
            if pred is None:
                return out
            # pred columns: x1,y1,x2,y2,conf,cls
            pred[:, :4] = self.scale_coords(im_t.shape[2:], pred[:, :4], img_rgb.shape).round()
            for *xyxy, conf, cls in pred.cpu().tolist():
                x1, y1, x2, y2 = map(int, xyxy)
                out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(conf), "cls": int(cls)})
            return out
        else:
            return []

# ----------------- Main Node -----------------
class LiveORBCompareNode(Node):
    def __init__(self):
        super().__init__("liveorb_gt_compare")
        self.get_logger().info("Starting LiveORB GT Comparative node")

        # params & topics
        self.declare_parameter("left_image_topic", LEFT_IMAGE_TOPIC)
        self.declare_parameter("right_image_topic", RIGHT_IMAGE_TOPIC)
        self.declare_parameter("odom_topic", ODOM_TOPIC)
        self.declare_parameter("gt_cones_topic", GT_CONES_TOPIC)
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
        self.gt_topic = self.get_parameter("gt_cones_topic").value

        base = Path(__file__).parent.resolve()
        repo_dir = base / self.get_parameter("yolo_repo").value
        weights_path = base / self.get_parameter("yolo_weights").value

        # intrinsics & baseline
        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)
        self.baseline = float(self.get_parameter("baseline").value)

        # YOLO
        try:
            self.yolo = YOLOv5Detector(repo_dir, weights_path, device="cpu")
        except Exception as e:
            self.get_logger().warning("YOLO failed to load: " + str(e))
            self.yolo = None

        # ORB
        self.orb = cv2.ORB_create(ORB_N_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # CV bridge
        self.bridge = CvBridge()
        self.left_img = None
        self.right_img = None

        # storage
        self.live_cones_world = []  # list of tuples (x_world, y_world, cls)
        self.gt_cones_world = []    # list of tuples (x_world, y_world, cls)
        self.odom_pose = (0.0, 0.0, 0.0)  # x,y,yaw

        # subscribers
        qos = qos_profile_sensor_data
        self.left_sub = self.create_subscription(RosImage, self.left_topic, self.left_cb, qos)
        self.right_sub = self.create_subscription(RosImage, self.right_topic, self.right_cb, qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos)

        ConesMsg = import_cones_msg()
        if ConesMsg is not None:
            self.gt_sub = self.create_subscription(ConesMsg, self.gt_topic, self.gt_cb, qos)
            self.get_logger().info(f"Subscribed to GT cones topic {self.gt_topic}")
        else:
            self.get_logger().warning("eufs_msgs ConeArray message not available; GT disabled")
            self.gt_sub = None

        # plotting thread
        self._stop = False
        self._plot_thread = threading.Thread(target=self.plotting_thread, daemon=True)
        self._plot_thread.start()

        # show windows update as separate thread to avoid blocking ROS callbacks
        self._cv_stop = False
        self._cv_thread = threading.Thread(target=self.cv_display_thread, daemon=True)
        self._cv_thread.start()

        self.get_logger().info("Node initialized")

    # ------------- Callbacks -------------
    def left_cb(self, msg: RosImage):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.left_img = cv_img
        except Exception as e:
            self.get_logger().error("left_cb cv_bridge error: " + str(e))

    def right_cb(self, msg: RosImage):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.right_img = cv_img
        except Exception as e:
            self.get_logger().error("right_cb cv_bridge error: " + str(e))

    def odom_cb(self, msg: Odometry):
        # extract yaw
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.odom_pose = (float(px), float(py), float(yaw))

    def gt_cb(self, msg):
        # parse GT cones list, convert to world coordinates if they are in car frame using odom
        # We will assume msg contains fields like yellow_cones, blue_cones, etc.
        # If the cones are already in world frame, then this will be world coordinates already.
        color_fields = [
            ('yellow_cones', 1),
            ('blue_cones', 0),
            ('orange_cones', 2),
            ('big_orange_cones', 3),
            ('unknown_color_cones', 4)
        ]
        parsed = []
        for field, cls in color_fields:
            cones = getattr(msg, field, [])
            for c in cones:
                x, y, z = xyz_from_cone(c)
                # Heuristic: many EUFS GT messages are in car frame (x forward,y left). The GT visualizer you provided plotted them directly.
                # Here we will store them as world coordinates by transforming with odom (so both GT and live are in world).
                ox, oy, oyaw = self.odom_pose
                # Transform car-frame GT to world frame: Xw = ox + x*cos(yaw) - y*sin(yaw)
                Xw = ox + x*math.cos(oyaw) - y*math.sin(oyaw)
                Yw = oy + x*math.sin(oyaw) + y*math.cos(oyaw)
                parsed.append((float(Xw), float(Yw), int(cls)))
        self.gt_cones_world = parsed

    # ------------- Stereo+ORB processing -------------
    def process_stereo(self):
        """
        Main processing: runs detection (if available) and ORB matching, then computes 3D points
        and transforms them to world frame.
        """
        if self.left_img is None or self.right_img is None:
            return

        imgL = self.left_img.copy()
        imgR = self.right_img.copy()

        # ---- YOLO detections (optional) ----
        boxesL = []
        boxesR = []
        if self.yolo is not None:
            try:
                detsL = self.yolo.infer(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
                detsR = self.yolo.infer(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
                boxesL = [(d['cls'], d['x1'], d['y1'], d['x2'], d['y2']) for d in detsL]
                boxesR = [(d['cls'], d['x1'], d['y1'], d['x2'], d['y2']) for d in detsR]
            except Exception as e:
                self.get_logger().warning("YOLO inference error: " + str(e))
                boxesL = []
                boxesR = []

        # ORB keypoints
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        kpL, desL = self.orb.detectAndCompute(grayL, None)
        kpR, desR = self.orb.detectAndCompute(grayR, None)
        if desL is None or desR is None:
            return

        matches_knn = self.bf.knnMatch(desL, desR, k=2)
        good = []
        for pair in matches_knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < LOWE_RATIO * n.distance:
                good.append(m)

        # If YOLO boxes are available, restrict to matches inside boxes and group by pairs
        matches_by_pair = {}  # key=((x1,y1,x2,y2),(x1',y1',x2',y2'),cls) -> list of ((xl,yl),(xr,yr))
        if boxesL and boxesR:
            for m in good:
                xl, yl = kpL[m.queryIdx].pt
                xr, yr = kpR[m.trainIdx].pt
                xl_i, yl_i, xr_i, yr_i = int(xl), int(yl), int(xr), int(yr)
                for clsL, x1L, y1L, x2L, y2L in boxesL:
                    if x1L <= xl_i <= x2L and y1L <= yl_i <= y2L:
                        for clsR, x1R, y1R, x2R, y2R in boxesR:
                            if clsL == clsR and x1R <= xr_i <= x2R and y1R <= yr_i <= y2R:
                                key = ((x1L,y1L,x2L,y2L),(x1R,y1R,x2R,y2R),clsL)
                                matches_by_pair.setdefault(key, []).append(((int(round(xl)), int(round(yl))), (int(round(xr)), int(round(yr)))))
                                break
                        break
        else:
            # No YOLO: try to group matches by horizontal proximity (coarse)
            # Build lists of clustered matches by right-u coordinate proximity
            for m in good:
                xl, yl = kpL[m.queryIdx].pt
                xr, yr = kpR[m.trainIdx].pt
                # find bin by rounding u coordinate of left
                found = False
                for key in matches_by_pair.keys():
                    # key shape: ((x1L..),(x1R..),cls)
                    pass
                # fallback: each match becomes its own pair (we'll average many matches)
                key = ((int(xl)-5,int(yl)-5,int(xl)+5,int(yl)+5),(int(xr)-5,int(yr)-5,int(xr)+5,int(yr)+5),-1)
                matches_by_pair.setdefault(key, []).append(((int(round(xl)), int(round(yl))), (int(round(xr)), int(round(yr)))))

        # For each box-pair compute average disparity -> triangulate
        live_world = []
        ox, oy, oyaw = self.odom_pose
        for (boxL, boxR, cls), pts in matches_by_pair.items():
            if len(pts) < MIN_MATCHES_PER_BOX:
                continue
            left_us = np.array([p[0][0] for p in pts], dtype=float)
            left_vs = np.array([p[0][1] for p in pts], dtype=float)
            right_us = np.array([p[1][0] for p in pts], dtype=float)
            # mean pixel coords
            uL = float(left_us.mean())
            vL = float(left_vs.mean())
            uR = float(right_us.mean())
            disparity = uL - uR
            if disparity <= 0.1:
                continue
            # triangulate in optical frame
            Z_cam = (self.fx * self.baseline) / disparity  # forward in meters (Z_cam)
            X_cam = (uL - self.cx) * Z_cam / self.fx       # right positive
            Y_cam = (vL - self.cy) * Z_cam / self.fy       # down positive

            # convert optical -> vehicle frame (x forward, y left, z up)
            X_vehicle = float(Z_cam)         # forward
            Y_vehicle = float(-X_cam)       # left
            Z_vehicle = float(-Y_cam)       # up

            # add camera position offset (camera at vehicle origin with height)
            # camera_x,y are zero per your input, so only vertical offset matters if used; BEV ignores height.
            # Transform vehicle -> world using odom pose
            Xw = ox + X_vehicle * math.cos(oyaw) - Y_vehicle * math.sin(oyaw)
            Yw = oy + X_vehicle * math.sin(oyaw) + Y_vehicle * math.cos(oyaw)
            live_world.append((float(Xw), float(Yw), int(cls)))

            # Also annotate boxes on images
            # (we will draw later in display thread using previously saved img copies)
        self.live_cones_world = live_world

    # ------------- Display threads -------------
    def cv_display_thread(self):
        """Continuously update OpenCV window for left/right with overlay"""
        window_name = "Stereo Left|Right"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while not self._cv_stop and rclpy.ok():
            if self.left_img is None or self.right_img is None:
                time.sleep(0.03)
                continue

            L = self.left_img.copy()
            R = self.right_img.copy()
            # draw live cones/world projections back onto left image for visual debugging
            # We will project world live_cones_world back to left camera for rough overlay:
            for (Xw, Yw, cls) in self.live_cones_world:
                # convert world -> vehicle frame (inverse transform)
                ox, oy, oyaw = self.odom_pose
                # vehicle coords:
                Xv = ( (Xw - ox) * math.cos(-oyaw) - (Yw - oy) * math.sin(-oyaw) )
                Yv = ( (Xw - ox) * math.sin(-oyaw) + (Yw - oy) * math.cos(-oyaw) )
                # vehicle -> camera optical
                Z_cam = Xv
                X_cam = -Yv
                Y_cam = - (CAMERA_Z - 0.0)  # approximate using camera_z; not accurate for reprojection; used only for small overlay
                # project to image
                if Z_cam <= 0.01:
                    continue
                u = int(round((X_cam * self.fx) / Z_cam + self.cx))
                v = int(round((Y_cam * self.fy) / Z_cam + self.cy))
                color = CLASS_COLORS_BGR.get(cls, (200,200,200))
                # draw circle if in bounds
                if 0 <= u < L.shape[1] and 0 <= v < L.shape[0]:
                    cv2.circle(L, (u,v), 6, color, 2)

            combined = np.hstack((L, R))
            cv2.imshow(window_name, combined)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                self._cv_stop = True
                break
        cv2.destroyAllWindows()

    def plotting_thread(self):
        """
        Matplotlib BEV loop showing:
         - GT cones (solid dots)
         - Live cones (hollow circles)
         - Car at odom position (origin of local plot)
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(8,6))
        while not self._stop and rclpy.ok():
            # update processing just-in-time
            try:
                self.process_stereo()
            except Exception as e:
                self.get_logger().debug("process_stereo exception: " + str(e))

            ax.clear()
            ax.set_title("Comparative BEV (GT = solid, LiveORB = hollow)")
            ax.set_xlabel("X forward (m)")
            ax.set_ylabel("Y left (m)")
            ax.set_xlim(BEV_XLIM)
            ax.set_ylim(BEV_YLIM)
            ax.grid(True, linestyle='--', alpha=0.4)
            # plot car origin marker at odom pose
            ox, oy, oyaw = self.odom_pose
            # We will plot relative to the current vehicle pose: center map at vehicle world pose.
            # For human-friendly visualization we plot GT and Live cones in vehicle-centered coordinates (translate by -odom)
            # So on plot, vehicle is at (0,0)
            ax.scatter([0.0], [0.0], s=80, facecolors='none', edgecolors='lime', linewidths=2, label='vehicle')

            # plot GT cones (transform to vehicle-centered coords)
            for (gx, gy, gcls) in self.gt_cones_world:
                # GT stored as world coords in gt_cb. Convert to vehicle-centered coords:
                # vehicle-centered = rotate and translate world -> vehicle
                # But in gt_cb we already transformed car->world. Now compute relative:
                rx = gx - ox
                ry = gy - oy
                # rotate by -yaw
                vx = rx * math.cos(-oyaw) - ry * math.sin(-oyaw)
                vy = rx * math.sin(-oyaw) + ry * math.cos(-oyaw)
                ax.scatter([vx], [vy], c=CLASS_COLORS_MPL.get(gcls, "0.6"), s=40, marker='o')

            # plot live cones (already world coords in live_cones_world)
            for (lx, ly, lcls) in self.live_cones_world:
                rx = lx - ox
                ry = ly - oy
                vx = rx * math.cos(-oyaw) - ry * math.sin(-oyaw)
                vy = rx * math.sin(-oyaw) + ry * math.cos(-oyaw)
                ax.scatter([vx], [vy], edgecolors=CLASS_COLORS_MPL.get(lcls, "tab:red"), facecolors='none', s=80, linewidths=1.5)

            ax.legend(loc='upper right', fontsize=8)
            fig.canvas.draw_idle()
            plt.pause(1.0 / max(1.0, BEV_HZ))

        plt.close(fig)

    def destroy_node(self):
        self._stop = True
        self._cv_stop = True
        super().destroy_node()

# ----------------- Entrypoint -----------------
def main(args=None):
    rclpy.init(args=args)
    node = LiveORBCompareNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
