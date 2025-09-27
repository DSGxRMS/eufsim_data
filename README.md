# EUFSIM Simulator API communication
Sample codes for the purpose of data collection and commands are organised in the repo. Add any templates or repeating package structure to the repository. Work in individual branches to avoid merge/pull/push conflicts.
DO NOT CHANGE SOMEONE ELSE's CODE WITHOUT DISCUSSION. IF DONE, FIX THE MERGE CONFLICTS!

## !!For git commands!!
<details>
<Summary>Click here to view</summary>

# Git Basics 

This guide covers the most common Git commands you’ll need for everyday work.

---

## Initialize a Repository and set URL
```
git init
```
and then
```
git add remote origin <url>
```
## Clone a repo
```
git clone <repo_name>
```
## Staging and Update
Add files to stage with
```
git add <filename>
```
*Use '.' if staging all*

```
git commit -m "Add Message Here"
```
If pushing for the first time, set upstream:
```
git push -u origin main
```
else use directly - ```git push```

To reset or remove files after ```git add .```
use
```
git reset
```
*To discard changes use ```git reset --hard <commit_id>```

## Checking past commits
```
git log --oneline
```
## Branching
Check branches with
```
git branch
```

Create branch with
```
git branch <name>
```

Switch to branch with

```
git checkout <name>
```

## Merging branch to another
```
git merge <branch>
```

*In case of merge conflict, resolve manually. Only in worst possible cases use ```git rebase``` - rolls back to the last saved commit
</details>

## Developed Packages
* Ground truth - gt_tap
* Lidar data - lidar_tap
* Controls Command - ctrl_tap


# General ROS commands for use of packages:

## Building a package

In the workspace directory, add a folder named ```"src"``` for the purpose of storing all packages
Shift into the working directory, and run on terminal:
```
ros2 pkg create <package_name> --build-type ament_python --dependencies rclpy <other_dependencies>
```

## Steps to remember after generating the boilerplate for the package:
* Ensure to only create code files in the subfolder named same as the package
* Ensure to configure dependencies in ```package.xml```
* Update the ```setup.py``` file and add the nodes in the console scripts list using the general scheme:
```
<node_name> = <package_name>.<node_file_python(without .py)>:main
```

## To launch the package, build the library first using
```
colcon build --symlink-install
```
*Note!: Ensure to build the package the parent directory of src. Use symlink to update and view changes in code live without having to rebuild packages, else ```colcon build``` works just fine*

# Running the packages
In a new ubuntu terminal, source the setup files
```
source install/setup.bash
```
This takes place in the folder containing the src directory.
For easier use, add the line in ```~/.bashrc``` -> allows sourcing the files automatically every time a new terminal is started

## To run the package
Use the following command
```
ros2 run <package_name> <node_name>
```

Note: Ensure the simulator is running, in case of errors, source the setup again or rebuild if unresolved.



# Some general commands usable in ros

### To find out the topics being published, use
```
ros2 topic list
```

### To find out avalaible packages, use
```
ros2 pkg executables 
```

For finding out the nodes of a specific topic :
```
ros2 pkg executables <package_name>
```

### To print out the topic data without having to write Ros node
```
ros2 topic echo /<topic_name>
```


# To launch the simulator
```
ros2 launch eufs_launcher eufs_launcher.launch.py
```
or
```
ros2 launch eufs_launcher simulation.launch.py \
  use_sim_time:=true \
  track:=skidpad \
  vehicleModel:=DynamicBicycle \
  commandMode:=acceleration \
  vehicleModelConfig:=configDry.yaml \
  robot_name:=eufs \
  gazebo_gui:=true \
  pub_ground_truth:=true \
  launch_group:=no_perception \
  rviz:=true
```


Use gazebo for testing and simulating, rqt interface for controls (manual or algorithmic).
Track selection available on the GUI


NOTE: Use ctrl+C to stop a running terminal or node!