# Camera feed collection nodes
To get the topics publishing, it is necessary to enable and align the gazebo based camera onto the racecar.
Since for testing purposes we are using the eufs model, we will eit the file using the following steps


## In the Ubuntu terminal
```
ROBOT_XACRO="$(ros2 pkg prefix eufs_racecar)/share/eufs_racecar/robots/eufs/robot.urdf.xacro"
```

### 1) backup
```
cp -a "$ROBOT_XACRO" "${ROBOT_XACRO}.bak"
```

### 2) open
```
nano "$ROBOT_XACRO"
```


Then paste the following code at the very end but within the ```<robot>``` container

```
<!-- ===== GAZEBO CAMERAS: LEFT / RIGHT (publish pixels) ===== -->
    <gazebo reference="zed_left_camera_frame">
        <sensor type="camera" name="zed_left_camera">
            <always_on>true</always_on>
            <update_rate>20</update_rate>
            <visualize>true</visualize>
            <camera>
            <horizontal_fov>1.3089969</horizontal_fov> <!-- ~75° -->
            <image>
                <width>1280</width>
                <height>720</height>
                <format>R8G8B8</format>
            </image>
            <clip><near>0.05</near><far>100.0</far></clip>
            </camera>
            <plugin name="gazebo_ros_camera_left" filename="libgazebo_ros_camera.so">
            <ros>
                <namespace>/</namespace>
                <remapping>image:=/camera_0/image_raw</remapping>
                <remapping>camera_info:=/camera_0/camera_info</remapping>
            </ros>
            <frame_name>zed_left_camera_optical_frame</frame_name>
            </plugin>
        </sensor>
    </gazebo>

    <gazebo reference="zed_right_camera_frame">
        <sensor type="camera" name="zed_right_camera">
            <always_on>true</always_on>
            <update_rate>20</update_rate>
            <visualize>true</visualize>
            <camera>
            <horizontal_fov>1.3089969</horizontal_fov>
            <image>
                <width>1280</width>
                <height>720</height>
                <format>R8G8B8</format>
            </image>
            <clip><near>0.05</near><far>100.0</far></clip>
            </camera>
            <plugin name="gazebo_ros_camera_right" filename="libgazebo_ros_camera.so">
            <ros>
                <namespace>/</namespace>
                <remapping>image:=/camera_1/image_raw</remapping>
                <remapping>camera_info:=/camera_1/camera_info</remapping>
            </ros>
            <frame_name>zed_right_camera_optical_frame</frame_name>
            </plugin>
        </sensor>
    </gazebo>
```


Use ```ctrl+X``` and then press "Y" to save. Restart or resource the terminal as well as teh simulator.

## Check Ros topics
```
ros2 topic list
```


# In Case of Plugin Error
```
sudo apt-get update
sudo apt-get install -y ros-galactic-gazebo-ros-pkgs ros-galactic-gazebo-plugins
```
Run the above installations first then run the nodes!




