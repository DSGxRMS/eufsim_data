# Controls Command Node for EUFSIM

The API works two ways depending upon the input mode selected in the Launcher GUI
Use 
```
ros2 run ctrl_tap command_acc
```
when selected acceleration input in GUI. Code is differentiated to take the accelerated logic

If instead, velocity is needed
run the other node ```command_vel```
