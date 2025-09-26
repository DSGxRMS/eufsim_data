import os
import math


def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def quat_to_yaw(w, x, y, z):
    # ZYX yaw from quaternion (ENU)
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def norm2(x, y): 
    return math.hypot(x, y)



