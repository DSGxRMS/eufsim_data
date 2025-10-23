import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage, CompressedImage as RosCompressedImage
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import random
from pathlib import Path

from bbox_data import YOLOv5Detector  # adjust import path as needed

# Custom message type definition might be needed for cone pairs, here we use ROS standard msgs for illustration:
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

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

        # Subscribers
        left_topic = self.get_parameter('left_image_topic').get_parameter_value().string_value
        right_topic = self.get_parameter('right_image_topic').get_parameter_value().string_value
        img_transport = self.get_parameter('image_transport').get_parameter_value().string_value

        qos_profile = rclpy.qos.QoSProfile(depth=10)

        if img_transport == 'compressed':
            from sensor_msgs.msg import CompressedImage
            self.left_sub = self.create_subscription(CompressedImage, left_topic, self.left_cb_compressed, qos_profile)
            self.right_sub = self.create_subscription(CompressedImage, right_topic, self.right_cb_compressed, qos_profile)
            self.get_logger().info('Subscribed to compressed stereo images')
        else:
            self.left_sub = self.create_subscription(RosImage, left_topic, self.left_cb_raw, qos_profile)
            self.right_sub = self.create_subscription(RosImage, right_topic, self.right_cb_raw, qos_profile)
            self.get_logger().info('Subscribed to raw stereo images')

        self.left_img = None
        self.right_img = None

        # Publisher for cone pairs (visualization markers here)
        self.marker_pub = self.create_publisher(MarkerArray, '/cone_pairs', 10)

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
        def inside(x,y,box): x1,y1,x2,y2=box; return x1<=x<=x2 and y1<=y<=y2
        matches_by_pair = {}

        for match in good_matches:
            xL,yL = map(int,kp_left[match.queryIdx].pt)
            xR,yR = map(int,kp_right[match.trainIdx].pt)
            for clsL,x1L,y1L,x2L,y2L in boxes_left:
                if inside(xL,yL,(x1L,y1L,x2L,y2L)):
                    for clsR,x1R,y1R,x2R,y2R in boxes_right:
                        if clsL==clsR and inside(xR,yR,(x1R,y1R,x2R,y2R)):
                            key = ((x1L,y1L,x2L,y2L),(x1R,y1R,x2R,y2R),clsL)
                            matches_by_pair.setdefault(key,[]).append(((xL,yL),(xR,yR)))
                            break
                    break

        # Draw and publish markers
        combined = np.hstack((self.left_img.copy(), self.right_img.copy()))
        offset_x = self.left_img.shape[1]
        distances = []
        markers = MarkerArray()
        marker_id = 0

        for pair_id,(key, points) in enumerate(matches_by_pair.items()):
            boxL, boxR, cls = key
            color = tuple(np.random.randint(0,255,3).tolist())
            color_norm = [c/255.0 for c in color]

            # Draw bounding boxes and matches
            cv2.rectangle(combined,(boxL[0],boxL[1]),(boxL[2],boxL[3]),color,2)
            cv2.rectangle(combined,(boxR[0]+offset_x,boxR[1]),(boxR[2]+offset_x,boxR[3]),color,2)

            disparities = []
            for (ptL,ptR) in points:
                xL,yL = ptL
                xR,yR = ptR
                disparity = abs(xL - xR)
                if disparity>0:
                    disparities.append(disparity)
                cv2.circle(combined,(xL,yL),3,color,-1)
                cv2.circle(combined,(xR+offset_x,yR),3,color,-1)
                cv2.line(combined,(xL,yL),(xR+offset_x,yR),color,1)

            if disparities:
                avg_disp = np.mean(disparities)
                distance_m = (self.focal_length * self.baseline) / avg_disp
                distances.append((pair_id+1, cls, distance_m))
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

            mid_x = (boxL[0]+boxL[2])//2
            label_pos = (mid_x, max(20, boxL[1]-10))
            cv2.putText(combined,f"Pair {pair_id+1}: {dist_text}",label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

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
