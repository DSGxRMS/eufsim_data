# Ground Truth API
* ```gt_pose``` — GT odometry: position (x,y,z), yaw, linear velocity, speed_xy.

* ```gt_state``` — Full car state blob from EUFS (all fields serialized).

* ```gt_track``` — Track layout/sections ground-truth message dump.

* ```gt_cones``` — Ground-truth cones (raw JSON) + flat per-cone (x,y,z,color).

* ```gt_wheels``` — Ground-truth wheel speeds (fl, fr, rl, rr).

* ```gt_gps``` — GPS fix: latitude, longitude, altitude, status.