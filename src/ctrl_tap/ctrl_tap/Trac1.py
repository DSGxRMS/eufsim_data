import time
import random

class TractionControl:
    def __init__(self, slip_threshold=0.1, control_gain=0.5):

        self.slip_threshold = slip_threshold
        self.control_gain = control_gain

    def compute_slip(self, wheel_speed, vehicle_speed):
    
        if vehicle_speed < 0.1:  
            return 0.0
        return (wheel_speed - vehicle_speed) / vehicle_speed

    def control(self, wheel_speeds, vehicle_speed, current_torque):
        
        # Applying traction control logic to adjust torque
        # wheel_speeds: list of wheel linear speeds (m/s)
        # vehicle_speed: IMU/GPS speed (m/s)
        # current_torque: current engine/motor torque request
       
        slips = [self.compute_slip(ws, vehicle_speed) for ws in wheel_speeds]
        max_slip = max(slips)

        if max_slip > self.slip_threshold:
            # Reducing torque proportional to slip excess
            torque_reduction = self.control_gain * (max_slip - self.slip_threshold)
            new_torque = current_torque * (1 - torque_reduction)
            new_torque = max(0, new_torque)  # no negative torque
            intervention = True
        else:
            new_torque = current_torque
            intervention = False

        return new_torque, slips, intervention


# ----------------- Example usage -----------------

if __name__ == "__main__":
    tc = TractionControl(slip_threshold=0.1, control_gain=0.5)

    # Simulated loop
    vehicle_speed = 20.0  # m/s from IMU/GPS
    current_torque = 200.0  # Nm

    for t in range(10):
        # Simulate wheel speed sensors (some slip)
        wheel_speeds = [
            vehicle_speed + random.uniform(-0.5, 3.0),  # front-left
            vehicle_speed + random.uniform(-0.5, 3.0),  # front-right
        ]

        new_torque, slips, intervention = tc.control(wheel_speeds, vehicle_speed, current_torque)

        print(f"Step {t}:")
        print(f"  Wheel speeds: {wheel_speeds}")
        print(f"  Vehicle speed: {vehicle_speed:.2f} m/s")
        print(f"  Slips: {[round(s,3) for s in slips]}")
        print(f"  Torque request: {new_torque:.2f} Nm (intervention={intervention})\n")

        time.sleep(1)