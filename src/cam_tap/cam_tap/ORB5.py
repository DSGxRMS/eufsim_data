import cv2
import numpy as np
import random
from pathlib import Path

# Import YOLOv5Detector from your bbox_data.py (adjust path as needed)
from bbox_data import YOLOv5Detector

# --- CONFIG ---
left_img_path = r"./dataset_bbox/left.jpg"
right_img_path = r"./dataset_bbox/right.jpg"
baseline = 0.12  # meters
focal_length_px = 700  # pixels

ORB_N_FEATURES = 2500
LOWE_RATIO = 0.8

# Initialize YOLOv5Detector with repo and weights paths
base_dir = Path(__file__).parent.resolve()
repo_dir = base_dir / "yolov5"                # Path to cloned YOLOv5 repo
weights_path = base_dir / "yolov5/weights/best.pt"  # Path to your YOLOv5 weights

yolo = YOLOv5Detector(repo_dir=repo_dir, weights_path=weights_path, device="cpu")  # or "cpu"

# Helper functions
def inside(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())

# Load stereo images
img_left = cv2.imread(left_img_path)
img_right = cv2.imread(right_img_path)
if img_left is None or img_right is None:
    raise FileNotFoundError("Image not found.")

# Convert images to RGB for YOLO detector
img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

# Run YOLOv5 inference to get detections
dets_left = yolo.infer(img_left_rgb)
dets_right = yolo.infer(img_right_rgb)

# Convert detections to box tuples (cls, x1, y1, x2, y2)
boxes_left = [(d["cls"], d["x1"], d["y1"], d["x2"], d["y2"]) for d in dets_left]
boxes_right = [(d["cls"], d["x1"], d["y1"], d["x2"], d["y2"]) for d in dets_right]

print(f"Left boxes: {len(boxes_left)} | Right boxes: {len(boxes_right)}")

# ORB feature extraction
orb = cv2.ORB_create(ORB_N_FEATURES)
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
kp_left, des_left = orb.detectAndCompute(gray_left, None)
kp_right, des_right = orb.detectAndCompute(gray_right, None)

# Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des_left, des_right, k=2)

# Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < LOWE_RATIO * n.distance:
        good_matches.append(m)

print(f"Total keypoints: Left={len(kp_left)}, Right={len(kp_right)} | Good matches={len(good_matches)}")

# Group matches by box pairs
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

print(f"Detected {len(matches_by_pair)} matching box pairs")

# Draw matches and estimate distances
combined = np.hstack((img_left.copy(), img_right.copy()))
offset_x = img_left.shape[1]
distances = []

for pair_id, (key, points) in enumerate(matches_by_pair.items()):
    (boxL, boxR, cls) = key
    color = random_color()

    # Draw bounding boxes
    cv2.rectangle(combined, (boxL[0], boxL[1]), (boxL[2], boxL[3]), color, 2)
    cv2.rectangle(combined, (boxR[0] + offset_x, boxR[1]), (boxR[2] + offset_x, boxR[3]), color, 2)

    disparities = []
    for (ptL, ptR) in points:
        xL, yL = ptL
        xR, yR = ptR
        disparity = abs(xL - xR)
        if disparity > 0:
            disparities.append(disparity)
        # Draw keypoints and connecting lines
        cv2.circle(combined, (xL, yL), 3, color, -1)
        cv2.circle(combined, (xR + offset_x, yR), 3, color, -1)
        cv2.line(combined, (xL, yL), (xR + offset_x, yR), color, 1)

    # Compute distance from disparity
    if len(disparities) > 0:
        avg_disp = np.mean(disparities)
        distance_m = (focal_length_px * baseline) / avg_disp
        distances.append((pair_id + 1, cls, distance_m))
        distance_text = f"{distance_m:.2f} m"
    else:
        distance_text = "No match"

    mid_x = (boxL[0] + boxL[2]) // 2
    label_pos = (mid_x, max(20, boxL[1] - 10))
    cv2.putText(combined, f"Pair {pair_id + 1}: {distance_text}", label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3)

# Print distances from nearest to farthest
distances.sort(key=lambda x: x[2])
print("\nCone Distances (nearest -> farthest):")
for pid, cls, dist in distances:
    print(f"  Pair {pid} (Class {cls}): {dist:.2f} m")

# Show result
cv2.imshow("ORB Matches + Distances", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
