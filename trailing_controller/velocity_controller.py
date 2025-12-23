import numpy as np


class VelocityController:
    def __init__(self,
                 desired_gap: float = 2.0,
                 min_gap: float = 1.0,
                 max_gap: float = 4.0,
                 kp: float = 0.8,
                 kd: float = 0.3,
                 ki: float = 0.05,
                 max_velocity: float = 7.0,
                 min_velocity: float = 0.5,
                 emergency_brake_distance: float = 0.8):
        self.desired_gap = desired_gap
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.emergency_brake_distance = emergency_brake_distance
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.max_integral = 2.0
        
    def compute_velocity(self, target_distance: float, target_vs: float, dt: float = 0.1) -> float:
        if target_distance < self.emergency_brake_distance:
            return 0.0
        
        gap_error = target_distance - self.desired_gap
        
        self.integral += gap_error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        
        derivative = (gap_error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = gap_error
        
        velocity_adjustment = (self.kp * gap_error + 
                             self.kd * derivative + 
                             self.ki * self.integral)
        
        base_velocity = max(target_vs, self.min_velocity)
        
        if target_distance < self.min_gap:
            desired_velocity = base_velocity * 0.5
        elif target_distance > self.max_gap:
            desired_velocity = self.max_velocity
        else:
            desired_velocity = base_velocity + velocity_adjustment
        
        desired_velocity = np.clip(desired_velocity, self.min_velocity, self.max_velocity)
        
        return desired_velocity
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
