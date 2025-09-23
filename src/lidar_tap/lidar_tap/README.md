# Lidar ROS Notes
Contained within the lidar_tap package providing the following functionality with example codes
* Subscriber model to the published LiDAR point cloud data
* LiDAR points viewer (3D visualiation) - displays reconstructed cones, centroids (red), ground removal (grey)
* benchmark tester to gauge fps without visualisation - currently heavy load, return $0.3$ FPS max on base algorithm



# Dependencies
```
python3 -m pip install --user PySide6 pyqtgraph scikit-learn pandas opencv-python
```