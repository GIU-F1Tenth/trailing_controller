import numpy as np


class StanleyController:
    def __init__(self,
                 k: float = 1.0,
                 ks: float = 0.5,
                 wheelbase: float = 0.33,
                 max_steer: float = 0.42):
        self.k = k
        self.ks = ks
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        
    def compute_steering(self, 
                        target_x: float, 
                        target_y: float,
                        ego_x: float = 0.0,
                        ego_y: float = 0.0,
                        ego_yaw: float = 0.0,
                        velocity: float = 1.0) -> float:
        
        dx = target_x - ego_x
        dy = target_y - ego_y
        target_yaw = np.arctan2(dy, dx)
        
        heading_error = self.normalize_angle(target_yaw - ego_yaw)
        
        front_axle_x = ego_x + self.wheelbase * np.cos(ego_yaw)
        front_axle_y = ego_y + self.wheelbase * np.sin(ego_yaw)
        
        dx_front = target_x - front_axle_x
        dy_front = target_y - front_axle_y
        
        cross_track_error = -dy_front * np.cos(target_yaw) + dx_front * np.sin(target_yaw)
        
        cross_track_term = np.arctan2(self.k * cross_track_error, self.ks + velocity)
        
        steering_angle = heading_error + cross_track_term
        
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        
        return steering_angle
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle


class PurePursuitController:
    def __init__(self,
                 lookahead_gain: float = 0.5,
                 lookahead_min: float = 1.0,
                 lookahead_max: float = 3.0,
                 wheelbase: float = 0.33,
                 max_steer: float = 0.42):
        self.lookahead_gain = lookahead_gain
        self.lookahead_min = lookahead_min
        self.lookahead_max = lookahead_max
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        
    def compute_steering(self,
                        target_x: float,
                        target_y: float,
                        ego_x: float = 0.0,
                        ego_y: float = 0.0,
                        ego_yaw: float = 0.0,
                        velocity: float = 1.0) -> float:
        
        lookahead_distance = np.clip(
            self.lookahead_gain * velocity,
            self.lookahead_min,
            self.lookahead_max
        )
        
        dx = target_x - ego_x
        dy = target_y - ego_y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 0.1:
            return 0.0
        
        scale = lookahead_distance / distance
        lookahead_x = ego_x + dx * scale
        lookahead_y = ego_y + dy * scale
        
        alpha = np.arctan2(lookahead_y - ego_y, lookahead_x - ego_x) - ego_yaw
        
        alpha = StanleyController.normalize_angle(alpha)
        
        steering_angle = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), lookahead_distance)
        
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        
        return steering_angle


class HybridSteeringController:
    def __init__(self,
                 wheelbase: float = 0.33,
                 max_steer: float = 0.42,
                 stanley_k: float = 1.0,
                 stanley_ks: float = 0.5,
                 pp_lookahead_gain: float = 0.5,
                 pp_lookahead_min: float = 1.0,
                 pp_lookahead_max: float = 3.0,
                 switch_velocity: float = 2.0):
        
        self.stanley = StanleyController(
            k=stanley_k,
            ks=stanley_ks,
            wheelbase=wheelbase,
            max_steer=max_steer
        )
        
        self.pure_pursuit = PurePursuitController(
            lookahead_gain=pp_lookahead_gain,
            lookahead_min=pp_lookahead_min,
            lookahead_max=pp_lookahead_max,
            wheelbase=wheelbase,
            max_steer=max_steer
        )
        
        self.switch_velocity = switch_velocity
        
    def compute_steering(self,
                        target_x: float,
                        target_y: float,
                        ego_x: float = 0.0,
                        ego_y: float = 0.0,
                        ego_yaw: float = 0.0,
                        velocity: float = 1.0) -> float:
        
        if velocity < self.switch_velocity:
            return self.stanley.compute_steering(
                target_x, target_y, ego_x, ego_y, ego_yaw, velocity
            )
        else:
            stanley_steer = self.stanley.compute_steering(
                target_x, target_y, ego_x, ego_y, ego_yaw, velocity
            )
            pp_steer = self.pure_pursuit.compute_steering(
                target_x, target_y, ego_x, ego_y, ego_yaw, velocity
            )
            
            blend_factor = min(1.0, (velocity - self.switch_velocity) / self.switch_velocity)
            return stanley_steer * (1.0 - blend_factor) + pp_steer * blend_factor
