#!/usr/bin/env python3

import numpy as np


class LateralController:
    def __init__(self, wheelbase=0.33, lookahead_gain=0.6, lookahead_offset=-0.18):
        self.wheelbase = wheelbase
        self.lookahead_gain = lookahead_gain
        self.lookahead_offset = lookahead_offset
        self.lookup_table = self._generate_lookup_table()
        
    def _generate_lookup_table(self):
        lookup = {}
        velocities = np.arange(0.5, 7.1, 0.1)
        steering_angles = np.concatenate([
            np.arange(0, 0.1, 0.0033),
            np.arange(0.1, 0.41, 0.01)
        ])
        
        for v in velocities:
            for delta in steering_angles:
                R = self.wheelbase / np.tan(delta) if np.abs(delta) > 1e-6 else 1e6
                a_lat = v**2 / R if R > 0 else 0
                lookup[(round(v, 1), round(delta, 4))] = a_lat
                
        return lookup
        
    def _get_steering_from_accel(self, v, a_lat_desired):
        v_rounded = round(v, 1)
        v_rounded = np.clip(v_rounded, 0.5, 7.0)
        
        best_delta = 0.0
        min_error = float('inf')
        
        steering_angles = np.concatenate([
            np.arange(0, 0.1, 0.0033),
            np.arange(0.1, 0.41, 0.01)
        ])
        
        for delta in steering_angles:
            key = (v_rounded, round(delta, 4))
            if key in self.lookup_table:
                a_lat = self.lookup_table[key]
                error = abs(a_lat - abs(a_lat_desired))
                if error < min_error:
                    min_error = error
                    best_delta = delta
                    
        return np.sign(a_lat_desired) * best_delta if abs(a_lat_desired) > 1e-6 else 0.0
        
    def compute_steering(self, ego_x, ego_y, ego_psi, ego_vx, target_x, target_y):
        L_d = self.lookahead_gain * ego_vx + self.lookahead_offset
        L_d = max(0.5, L_d)
        
        dx = target_x - ego_x
        dy = target_y - ego_y
        
        alpha = np.arctan2(dy, dx) - ego_psi
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        eta = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        a_c = 2 * ego_vx**2 * np.sin(eta) / L_d if L_d > 0 else 0.0
        
        steering_angle = self._get_steering_from_accel(ego_vx, a_c)
        
        return steering_angle
