#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from typing import Tuple, Optional


class FrenetConverter:
    def __init__(self, raceline_s: Optional[np.ndarray] = None,
                 raceline_x: Optional[np.ndarray] = None,
                 raceline_y: Optional[np.ndarray] = None):
        self.raceline_s = raceline_s
        self.raceline_x = raceline_x
        self.raceline_y = raceline_y
        
        if raceline_s is not None and raceline_x is not None:
            self.interp_x = interp1d(raceline_s, raceline_x, kind='cubic', 
                                     fill_value='extrapolate')
            self.interp_y = interp1d(raceline_s, raceline_y, kind='cubic',
                                     fill_value='extrapolate')
        else:
            self.interp_x = None
            self.interp_y = None
            
    def update_raceline(self, raceline_s: np.ndarray, raceline_x: np.ndarray,
                        raceline_y: np.ndarray):
        self.raceline_s = raceline_s
        self.raceline_x = raceline_x
        self.raceline_y = raceline_y
        
        self.interp_x = interp1d(raceline_s, raceline_x, kind='cubic',
                                fill_value='extrapolate')
        self.interp_y = interp1d(raceline_s, raceline_y, kind='cubic',
                                fill_value='extrapolate')
        
    def cartesian_to_frenet(self, x: float, y: float) -> Tuple[float, float]:
        if self.raceline_s is None:
            return 0.0, np.sqrt(x**2 + y**2)
            
        raceline_points = np.column_stack((self.raceline_x, self.raceline_y))
        point = np.array([[x, y]])
        
        distances = cdist(point, raceline_points)[0]
        closest_idx = np.argmin(distances)
        
        s = self.raceline_s[closest_idx]
        
        if closest_idx < len(self.raceline_s) - 1:
            p1 = raceline_points[closest_idx]
            p2 = raceline_points[closest_idx + 1]
        else:
            p1 = raceline_points[closest_idx - 1]
            p2 = raceline_points[closest_idx]
            
        tangent = p2 - p1
        tangent = tangent / np.linalg.norm(tangent)
        
        normal = np.array([-tangent[1], tangent[0]])
        
        vector_to_point = point[0] - raceline_points[closest_idx]
        d = np.dot(vector_to_point, normal)
        
        return s, d
        
    def frenet_to_cartesian(self, s: float, d: float) -> Tuple[float, float]:
        if self.interp_x is None:
            return 0.0, 0.0
            
        x_center = float(self.interp_x(s))
        y_center = float(self.interp_y(s))
        
        ds = 0.01
        x_ahead = float(self.interp_x(s + ds))
        y_ahead = float(self.interp_y(s + ds))
        
        tangent = np.array([x_ahead - x_center, y_ahead - y_center])
        tangent = tangent / np.linalg.norm(tangent)
        
        normal = np.array([-tangent[1], tangent[0]])
        
        x = x_center + d * normal[0]
        y = y_center + d * normal[1]
        
        return x, y
    
    def frenet_velocity_to_cartesian(self, s: float, d: float, vs: float, vd: float) -> Tuple[float, float]:
        if self.interp_x is None:
            return 0.0, 0.0
        
        ds = 0.01
        x_center = float(self.interp_x(s))
        y_center = float(self.interp_y(s))
        x_ahead = float(self.interp_x(s + ds))
        y_ahead = float(self.interp_y(s + ds))
        
        tangent = np.array([x_ahead - x_center, y_ahead - y_center])
        tangent = tangent / np.linalg.norm(tangent)
        
        normal = np.array([-tangent[1], tangent[0]])
        
        vx = vs * tangent[0] + vd * normal[0]
        vy = vs * tangent[1] + vd * normal[1]
        
        return float(vx), float(vy)
    
    def cartesian_velocity_to_frenet(self, s: float, d: float, vx: float, vy: float) -> Tuple[float, float]:
        if self.interp_x is None:
            return 0.0, 0.0
        
        ds = 0.01
        x_center = float(self.interp_x(s))
        y_center = float(self.interp_y(s))
        x_ahead = float(self.interp_x(s + ds))
        y_ahead = float(self.interp_y(s + ds))
        
        tangent = np.array([x_ahead - x_center, y_ahead - y_center])
        tangent = tangent / np.linalg.norm(tangent)
        
        normal = np.array([-tangent[1], tangent[0]])
        
        v_cartesian = np.array([vx, vy])
        vs = np.dot(v_cartesian, tangent)
        vd = np.dot(v_cartesian, normal)
        
        return float(vs), float(vd)