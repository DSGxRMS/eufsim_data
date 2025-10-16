#!/usr/bin/env python3
import os, io, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CompressedImage as RosCompressedImage

# =============================
#      Small helpers
# =============================
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


# =============================
#       Image rate helper
# =============================
@dataclass
class ImageWindow:
    num_images: int
    window_sec: float = 5.0
    saved: int = 0
    next_t: float = 0.0
    start_t: float = 0.0

    def reset_window(self, now: float):
        self.start_t = now
        self.saved = 0
        self.next_t = now  # save first frame asap

    def should_save(self, now: float) -> bool:
        if self.num_images <= 0:
            return False
        if self.saved >= self.num_images:
            if now - self.start_t >= self.window_sec:
                self.reset_window(now)
            return False
        interval = self.window_sec / float(self.num_images)
        if now >= self.next_t:
            self.next_t = now + interval
            return True
        return False


# =============================
#         Main Node
# =============================
class FixedRouteDriver(Node):
    def __init__(self):
        super().__init__("fixed_route_driver", automatically_declare_parameters_from_overrides=True)

        # -------- Params --------
        self.declare_parameter("wheelbase_m", 1.5)
        self.declare_parameter("radius_m", 9.125)
        self.declare_parameter("target_speed_mps", 4.0)
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
        self.declare_parameter("image_transport", "raw")   # "raw" or "compressed"
        self.declare_parameter("num_images", 100)          # per window_seconds
        self.declare_parameter("window_seconds", 5.0)

        self.declare_parameter("sync_slop_ms", 120.0)      # pairing tolerance
        self.declare_parameter("qos_best_effort", True)
        self.declare_parameter("rb_max", 300)              # ring buffer size per side

        gp = self.get_parameter
        f = lambda k: float(getattr(gp(k).get_parameter_value(), "double_value",
                                    gp(k).get_parameter_value().integer_value))
        i = lambda k: int(getattr(gp(k).get_parameter_value(), "integer_value",
                                   gp(k).get_parameter_value().double_value))
        s = lambda k: gp(k).get_parameter_value().string_value
        b = lambda k: gp(k).get_parameter_value().bool_value

        self.wheelbase       = f("wheelbase_m")
        self.radius          = f("radius_m")
        self.v_target        = f("target_speed_mps")
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
        self.num_images      = i("num_images")
        self.window_seconds  = f("window_seconds")
        self.sync_slop_ms    = f("sync_slop_ms")
        self.qos_best_effort = b("qos_best_effort")
        self.rb_max          = i("rb_max")

        # -------- Image dirs --------
        base = Path(__file__).parent
        self.left_dir  = (base / "dataset" / "left_cam");  self.left_dir.mkdir(parents=True, exist_ok=True)
        self.right_dir = (base / "dataset" / "right_cam"); self.right_dir.mkdir(parents=True, exist_ok=True)

        # -------- Live write + pairing state --------
        self.left_buf:  List[Tuple[int, bytes]] = []   # (stamp_ns, jpeg_bytes)
        self.right_buf: List[Tuple[int, bytes]] = []
        self.collect_enabled = True

        # Shared rate limiter (wall time)
        self.pair_win = ImageWindow(self.num_images, self.window_seconds)
        self.pair_win.reset_window(time.time())

        # -------- State machine --------
        self.state = "WAIT_INIT"
        self.p0 = (0.0, 0.0); self.yaw0 = 0.0; self.fwd0 = (1.0, 0.0); self.s1_progress = 0.0
        self.p1 = (0.0, 0.0); self.yaw1 = 0.0; self.center = (0.0, 0.0); self.theta_start = 0.0; self.theta_progress = 0.0
        self.p2 = (0.0, 0.0); self.yaw2 = 0.0; self.fwd2 = (1.0, 0.0); self.s2_progress = 0.0

        # odom
        self.have_odom = False
        self.pos = (0.0, 0.0); self.yaw = 0.0; self.speed = 0.0

        # fixed circle steer
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

        self._start_time = time.time()
        self.timer = self.create_timer(1.0 / max(1.0, self.control_hz), self._tick)

        # debug counters
        self._pairs_saved = 0
        self._left_seen = 0
        self._right_seen = 0
        self._last_log_t = time.time()

    # -------- Callbacks --------
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position; q = msg.pose.pose.orientation
        self.pos = (float(p.x), float(p.y))
        self.yaw = yaw_from_quat(q.w, q.x, q.y, q.z)
        vx = float(msg.twist.twist.linear.x); vy = float(msg.twist.twist.linear.y)
        self.speed = math.hypot(vx, vy)
        self.have_odom = True

    # -------- Publish helper (ACCEL-FIRST) --------
    def _publish_cmd(self, accel: float, steering: float):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        msg.drive.acceleration   = float(accel)
        msg.drive.speed          = 0.0
        self.pub_ack.publish(msg)

    # -------- State machine control --------
    def _tick(self):
        if not self.have_odom:
            self._publish_cmd(accel=min(self.accel_limit, 0.5*self.accel_limit), steering=0.0)
            return

        steering_cmd = 0.0

        if self.state == "WAIT_INIT":
            self.p0 = self.pos; self.yaw0 = self.yaw; self.fwd0 = unit_vec_from_yaw(self.yaw0)
            self.s1_progress = 0.0
            self.state = "STRAIGHT1"
            self.get_logger().info("[state] STRAIGHT1 start")

        elif self.state == "STRAIGHT1":
            d = (self.pos[0] - self.p0[0], self.pos[1] - self.p0[1])
            self.s1_progress = dot(d, self.fwd0)
            steering_cmd = 0.0
            if self.s1_progress >= self.seg1_len:
                self.p1 = self.pos; self.yaw1 = self.yaw
                n_hat = right_normal_from_yaw(self.yaw1) if self.right_sign < 0 else left_normal_from_yaw(self.yaw1)
                self.center = (self.p1[0] + self.radius * n_hat[0], self.p1[1] + self.radius * n_hat[1])
                self.theta_start = math.atan2(self.p1[1] - self.center[1], self.p1[0] - self.center[0])
                self.theta_progress = 0.0
                self.state = "CIRCLE"
                self.get_logger().info(f"[state] CIRCLE start center={self.center} theta0={self.theta_start:.3f}")

        elif self.state == "CIRCLE":
            steering_cmd = self.delta_circle
            theta_now = math.atan2(self.pos[1] - self.center[1], self.pos[0] - self.center[0])
            if self.right_sign < 0:
                self.theta_progress = wrap_to_2pi(self.theta_start - theta_now)
            else:
                self.theta_progress = wrap_to_2pi(theta_now - self.theta_start)
            if self.theta_progress >= (2.0 * math.pi - 1e-2):
                self.p2 = self.pos; self.yaw2 = self.yaw; self.fwd2 = unit_vec_from_yaw(self.yaw2)
                self.s2_progress = 0.0
                self.state = "STRAIGHT2"
                self.get_logger().info("[state] STRAIGHT2 start")

        elif self.state == "STRAIGHT2":
            steering_cmd = 0.0
            d = (self.pos[0] - self.p2[0], self.pos[1] - self.p2[1])
            self.s2_progress = dot(d, self.fwd2)
            if self.s2_progress >= self.seg2_len:
                self.state = "DONE"
                self.get_logger().info("[state] DONE; braking to stop")
                self.collect_enabled = False  # stop grabbing immediately

        else:  # DONE
            v_err = 0.0 - self.speed
            accel_cmd = max(-self.accel_limit, min(self.accel_limit, self.accel_gain * v_err))
            self._publish_cmd(accel=accel_cmd, steering=0.0)
            return

        # accel control to reach/hold v_target
        v_err = self.v_target - self.speed
        accel_cmd = max(-self.accel_limit, min(self.accel_limit, self.accel_gain * v_err))
        self._publish_cmd(accel=accel_cmd, steering=steering_cmd)

        # periodic debug
        now = time.time()
        if now - self._last_log_t >= 2.0:
            self.get_logger().info(
                f"[img] L_seen={self._left_seen} R_seen={self._right_seen} pairs_saved={self._pairs_saved} "
                f"Lbuf={len(self.left_buf)} Rbuf={len(self.right_buf)} collecting={self.collect_enabled}"
            )
            self._last_log_t = now

    # ======== Plain image callbacks + pairing (RAW) ========
    def _left_raw_cb(self, msg: RosImage):
        if not self.collect_enabled: return
        self._left_seen += 1
        stamp = self._stamp_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.left_buf.append((stamp, self._jpeg_bytes_from_raw(msg)))
        self._prune_buffers()
        self._try_pair_from_left(stamp)

    def _right_raw_cb(self, msg: RosImage):
        if not self.collect_enabled: return
        self._right_seen += 1
        stamp = self._stamp_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.right_buf.append((stamp, self._jpeg_bytes_from_raw(msg)))
        self._prune_buffers()
        self._try_pair_from_right(stamp)

    # ======== Plain image callbacks + pairing (COMPRESSED) ========
    def _left_compressed_cb(self, msg: RosCompressedImage):
        if not self.collect_enabled: return
        self._left_seen += 1
        stamp = self._stamp_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.left_buf.append((stamp, self._jpeg_bytes_from_compressed(msg)))
        self._prune_buffers()
        self._try_pair_from_left(stamp)

    def _right_compressed_cb(self, msg: RosCompressedImage):
        if not self.collect_enabled: return
        self._right_seen += 1
        stamp = self._stamp_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.right_buf.append((stamp, self._jpeg_bytes_from_compressed(msg)))
        self._prune_buffers()
        self._try_pair_from_right(stamp)

    # ======== Pairing helpers (LIVE WRITE) ========
    def _try_pair_from_left(self, stamp_left: int):
        if not self.right_buf: return
        idx, diff = self._nearest_index(self.right_buf, stamp_left)
        if idx is None or diff > self._slop_ns(): return
        if not self.pair_win.should_save(time.time()): return

        base = str(stamp_left)
        l_bytes = self.left_buf[-1][1] if self.left_buf and self.left_buf[-1][0] == stamp_left \
                  else self._take_latest(self.left_buf, stamp_left)[1]
        r_bytes = self.right_buf.pop(idx)[1]
        self._write_pair(base, l_bytes, r_bytes)

    def _try_pair_from_right(self, stamp_right: int):
        if not self.left_buf: return
        idx, diff = self._nearest_index(self.left_buf, stamp_right)
        if idx is None or diff > self._slop_ns(): return
        if not self.pair_win.should_save(time.time()): return

        base = str(stamp_right)
        r_bytes = self.right_buf[-1][1] if self.right_buf and self.right_buf[-1][0] == stamp_right \
                  else self._take_latest(self.right_buf, stamp_right)[1]
        l_bytes = self.left_buf.pop(idx)[1]
        self._write_pair(base, l_bytes, r_bytes)

    def _write_pair(self, base: str, l_bytes: bytes, r_bytes: bytes):
        # atomic writes so you never see half files
        l_path = self.left_dir / f"L_{base}.jpg"
        r_path = self.right_dir / f"R_{base}.jpg"
        self._atomic_write(l_path, l_bytes)
        self._atomic_write(r_path, r_bytes)
        self._pairs_saved += 1
        self.pair_win.saved += 1
        # quick log so you can see it happening live
        self.get_logger().info(f"[img] saved pair #{self._pairs_saved} -> {base}")

    @staticmethod
    def _atomic_write(path: Path, data: bytes):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # atomic rename

    # ======== utils ========
    def _nearest_index(self, buf: List[Tuple[int, bytes]], target: int):
        if not buf: return (None, None)
        best_i, best_d = None, 1<<62
        for i in range(len(buf)-1, -1, -1):
            d = abs(buf[i][0] - target)
            if d < best_d:
                best_d = d; best_i = i
            if buf[i][0] < target and best_i is not None and d > best_d:
                break
        return (best_i, best_d)

    def _take_latest(self, buf: List[Tuple[int, bytes]], stamp: int):
        for i in range(len(buf)-1, -1, -1):
            if buf[i][0] == stamp:
                return buf[i]
        return buf[-1]

    def _slop_ns(self) -> int:
        return int(self.sync_slop_ms * 1e6)

    def _prune_buffers(self):
        if len(self.left_buf)  > self.rb_max:  del self.left_buf[:len(self.left_buf)-self.rb_max]
        if len(self.right_buf) > self.rb_max:  del self.right_buf[:len(self.right_buf)-self.rb_max]

    @staticmethod
    def _stamp_ns(sec: int, nsec: int) -> int:
        return int(sec) * 10**9 + int(nsec)

    def _jpeg_bytes_from_raw(self, msg: RosImage) -> bytes:
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
            pil = PILImage.fromarray(arr, mode="RGB")

        elif enc in ("mono8", "8uc1"):
            if msg.step == w:
                arr = data.reshape(h, w)
            else:
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, msg.step)[:, :w]
            pil = PILImage.fromarray(arr, mode="L").convert("RGB")

        else:
            expected = w * 3
            arr = data.reshape(h, msg.step)[:, :expected].reshape(h, w, 3)
            pil = PILImage.fromarray(arr, mode="RGB")

        bio = io.BytesIO()
        pil.save(bio, format="JPEG", quality=92)
        return bio.getvalue()

    def _jpeg_bytes_from_compressed(self, msg: RosCompressedImage) -> bytes:
        bio_in = io.BytesIO(bytes(msg.data))
        pil = PILImage.open(bio_in).convert("RGB")
        bio_out = io.BytesIO()
        pil.save(bio_out, format="JPEG", quality=92)
        return bio_out.getvalue()


def main():
    rclpy.init()
    node = FixedRouteDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.collect_enabled = False  # stop grabs
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
