#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, math, json
from dataclasses import is_dataclass, asdict

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def csv_logger(csv_path, fieldnames):
    """Return a writer function that appends rows to CSV with header-once."""
    ensure_dir(os.path.dirname(csv_path))
    header_written = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    f = open(csv_path, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if not header_written:
        w.writeheader()
    def write(row: dict):
        w.writerow(row); f.flush()
    return write

def quat_to_yaw(w, x, y, z):
    # ZYX yaw from quaternion (ENU)
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def norm2(x, y): return math.hypot(x, y)

def rosmsg_to_native(obj):
    """Recursively convert ROS messages to Python types for JSON storage."""
    # Handle ROS messages (having __slots__ and _get_types)
    if hasattr(obj, '__slots__') and hasattr(obj, '_get_types'):
        d = {}
        for slot in obj.__slots__:
            v = getattr(obj, slot)
            d[slot] = rosmsg_to_native(v)
        return d
    # Dataclasses
    if is_dataclass(obj):
        return rosmsg_to_native(asdict(obj))
    # Lists / tuples
    if isinstance(obj, (list, tuple)):
        return [rosmsg_to_native(v) for v in obj]
    # Dicts
    if isinstance(obj, dict):
        return {k: rosmsg_to_native(v) for k, v in obj.items()}
    # Builtins
    return obj

def json_str(obj):
    try:
        return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)
    except Exception:
        return str(obj)
