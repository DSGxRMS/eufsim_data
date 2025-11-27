#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import Odometry
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import threading
from pathlib import Path
import math
import torch
from collections import Counter

# ----------------- CONFIGURATION -----------------
FX = 448.1338
BASELINE = 0.12
CX = 640.5

# Mapping Logic
MERGE_RADIUS = 1.2      # Radius to consider two detections as the same cone
MIN_CONFIDENCE = 5      # Need 5 detections to confirm a cone
MAX_DIST = 25.0         # Ignore cones further than 25m (too noisy)

# Colors
COLORS = {0: "gold", 1: "blue", 2: "darkorange", 3: "red", 4: "gray"}

# ----------------- STABLE CONE CLASS -----------------
class StableCone:
    def __init__(self, x, y, cls_id):
        self.x = x
        self.y = y
        # We track color votes (if YOLO flickers Blue/Yellow, we take majority)
        self.cls_votes = Counter()
        self.cls_votes[cls_id] += 1
        
        # Weight Accumulator (Inverse Variance)
        # We initialize with a weight based on initial distance
        w = self.calculate_weight(x)
        self.sum_x = x * w
        self.sum_y = y * w
        self.sum_w = w
        
        self.count = 1
        self.last_seen = 0 # frames since seen

    def calculate_weight(self, dist):
        # Inverse Variance Weighting
        # Error grows with dist^2. Weight should be 1/dist^2.
        # Close (2m) -> Weight 0.25
        # Far (20m) -> Weight 0.0025
        # This makes close measurements 100x more important than far ones.
        return 1.0 / (dist**2 + 0.1)

    def update(self, new_x, new_y, new_cls):
        w = self.calculate_weight(new_x)
        
        # Weighted Average Update
        self.sum_x += new_x * w
        self.sum_y += new_y * w
        self.sum_w += w
        
        self.x = self.sum_x / self.sum_w
        self.y = self.sum_y / self.sum_w
        
        self.cls_votes[new_cls] += 1
        self.count += 1
        self.last_seen = 0

    def apply_odom(self, dx, dy, dyaw):
        # Rotate and Translate this cone based on car motion
        # This keeps the cone "pinned" to the world while the car moves
        c = math.cos(dyaw)
        s = math.sin(dyaw)
        
        # Translate
        self.x -= dx
        self.y -= dy
        
        # Rotate (Active Rotation)
        nx = self.x * c + self.y * s
        ny = -self.x * s + self.y * c
        
        # Update accumulators so the math holds for next update
        self.x = nx
        self.y = ny
        # Reset weighted sums to current position to prevent drift history from dragging it back
        self.sum_x = self.x * self.sum_w
        self.sum_y = self.y * self.sum_w

    @property
    def best_cls(self):
        return self.cls_votes.most_common(1)[0][0]

# ----------------- YOLO -----------------
class YOLOv5:
    def __init__(self, repo, weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = torch.hub.load(str(repo), "custom", path=str(weights), source="local")
            self.model.to(self.device).eval()
            self.ok = True
        except:
            self.ok = False

    def infer(self, img):
        if not self.ok:
            return []
        res = self.model(img, size=640)
        dets = []
        if len(res.xyxy) > 0:
            for x1, y1, x2, y2, conf, cls in res.xyxy[0].cpu().numpy():
                if conf > 0.50:
                    dets.append((int(x1), int(y1), int(x2), int(y2), int(cls)))
        return dets

def get_gt_pt(cone):
    p = getattr(cone, 'location', getattr(cone, 'position', getattr(cone, 'point', None)))
    return (p.x, p.y) if p else (0.0, 0.0)

# ----------------- MAIN NODE -----------------
class StableMapperNode(Node):
    def __init__(self):
        super().__init__('stable_mapper')
        
        base = Path(__file__).parent.resolve()
        self.yolo = YOLOv5(base / "yolov5", base / "yolov5/weights/best.pt")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        self.last_img_l = None
        self.last_img_r = None
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        
        # Odom State
        self.last_odom = None
        self.current_odom = None # (x, y, yaw)
        
        self.gt_cones = []
        self.cones = [] # List of StableCone objects

        self.create_subscription(RosImage, "/zed/left/image_rect_color", self.cb_left, qos_profile_sensor_data)
        self.create_subscription(RosImage, "/zed/right/image_rect_color", self.cb_right, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/ground_truth/odom", self.cb_odom, qos_profile_sensor_data)
        try:
            from eufs_msgs.msg import ConeArrayWithCovariance
            self.create_subscription(ConeArrayWithCovariance, "/ground_truth/cones", self.cb_gt, qos_profile_sensor_data)
        except:
            pass

        self.create_timer(0.1, self.process_pipeline)

        matplotlib.use("Qt5Agg")
        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        self.timer_plot = self.create_timer(0.1, self.update_plot)
        self.get_logger().info("Stable Mapper Node Running.")

    def cb_left(self, msg):
        self.last_img_l = msg

    def cb_right(self, msg):
        self.last_img_r = msg

    def cb_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        with self.lock:
            self.current_odom = (p.x, p.y, yaw)

    def cb_gt(self, msg):
        temp = []
        # Map EUFS fields → class IDs that match COLORS:
        # 0: yellow, 1: blue, 2: orange, 3: big orange
        sources = [
            (msg.yellow_cones, 0),       # yellow
            (msg.blue_cones, 1),         # blue
            (msg.orange_cones, 2),       # orange
            (msg.big_orange_cones, 3),   # big orange
        ]
        for cones, cid in sources:
            for c in cones:
                x, y = get_gt_pt(c)
                temp.append((x, y, cid))
        with self.lock:
            self.gt_cones = temp

    def apply_motion_compensation(self):
        if self.last_odom is None or self.current_odom is None:
            return
        x1, y1, yaw1 = self.last_odom
        x2, y2, yaw2 = self.current_odom
        
        # Calculate Delta in World Frame
        dx_w = x2 - x1
        dy_w = y2 - y1
        dyaw = yaw2 - yaw1
        
        # Rotate Delta into Previous Car Frame
        # We need to know how much the car moved *relative to its own nose*
        c = math.cos(yaw1)
        s = math.sin(yaw1)
        
        forward = dx_w * c + dy_w * s
        side = -dx_w * s + dy_w * c
        
        # Apply to all cones
        for cone in self.cones:
            cone.apply_odom(forward, side, dyaw)

    def process_pipeline(self):
        if self.last_img_l is None or self.last_img_r is None:
            return
        
        # 1. Update Odom Physics
        with self.lock:
            self.apply_motion_compensation()
            if self.current_odom:
                self.last_odom = self.current_odom

        try:
            imgL_color = self.bridge.imgmsg_to_cv2(self.last_img_l, "bgr8")
            imgL = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2GRAY)
            imgR = self.bridge.imgmsg_to_cv2(self.last_img_r, "mono8")
        except:
            return

        # 2. Vision
        disp_map = self.stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        rgb = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2RGB)
        boxes = self.yolo.infer(rgb)
        
        new_measurements = []
        
        for x1, y1, x2, y2, cls_id in boxes:
            if x1 < 0 or y2 > disp_map.shape[0] or x2 > disp_map.shape[1]:
                continue
            
            roi = disp_map[y1:y2, x1:x2]
            valid = roi[roi > 1.0]
            if len(valid) < 10:
                continue
            
            disp = np.median(valid)
            if 2.0 < disp < 150.0:
                Z = (FX * BASELINE) / disp
                X = ((x1 + x2)/2 - CX) * Z / FX
                
                x_car = Z
                y_car = -X
                
                # Loose Range Filter
                if 1.0 < x_car < MAX_DIST and abs(y_car) < 12.0:
                    new_measurements.append((x_car, y_car, cls_id))

        self.update_map(new_measurements)

    def update_map(self, measurements):
        with self.lock:
            # 1. Update Existing Cones (Greedy Association)
            used_meas = set()
            
            for cone in self.cones:
                cone.last_seen += 1 # Age it
                
                best_dist = MERGE_RADIUS
                best_idx = -1
                
                for i, (mx, my, mcls) in enumerate(measurements):
                    if i in used_meas:
                        continue
                    dist = math.hypot(cone.x - mx, cone.y - my)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                
                if best_idx != -1:
                    # Found match! Update cone position (Weighted Average)
                    mx, my, mcls = measurements[best_idx]
                    cone.update(mx, my, mcls)
                    used_meas.add(best_idx)

            # 2. Add New Cones
            for i, (mx, my, mcls) in enumerate(measurements):
                if i not in used_meas:
                    self.cones.append(StableCone(mx, my, mcls))

            # 3. Prune Map
            # Remove cones that haven't been seen recently
            # Keep them longer if they were seen many times (Confidence)
            self.cones = [c for c in self.cones if c.last_seen < (20 if c.count > 10 else 5)]

    def update_plot(self):
        with self.lock:
            map_data = list(self.cones)
            gt = list(self.gt_cones)

        self.ax.cla()
        self.ax.set_title(f"Stable Map (Cones: {len(map_data)})")
        self.ax.set_xlim(8, -8)   
        self.ax.set_ylim(-2, 22)  
        self.ax.grid(True, alpha=0.3)
        self.ax.plot(0, 0, '^', color='lime', ms=12, label='Ego')

        # GT cones: all, hollow, correct colors
        for x, y, c in gt:
            col = COLORS.get(c, 'black')
            self.ax.plot(y, x, 'o', color=col, fillstyle='none', ms=10, mew=2)

        # Algo cones: only confirmed, solid, correct colors
        for c in map_data:
            if c.count < MIN_CONFIDENCE:
                continue
            
            x, y = c.x, c.y
            col = COLORS.get(c.best_cls, 'black')
            self.ax.plot(y, x, 'o', color=col, ms=7)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

def main():
    rclpy.init()
    try:
        rclpy.spin(StableMapperNode())
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
