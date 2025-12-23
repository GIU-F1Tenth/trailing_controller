#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import CubicSpline


class FrenetConverter:
    def __init__(self):
        self.raceline_s = None
        self.raceline_x = None
        self.raceline_y = None
        self.raceline_psi = None
        self.x_spline = None
        self.y_spline = None
        self.psi_spline = None
        
    def update_raceline(self, s_coords, x_coords, y_coords):
        self.raceline_s = s_coords
        self.raceline_x = x_coords
        self.raceline_y = y_coords
        
        psi = np.zeros(len(s_coords))
        for i in range(len(s_coords) - 1):
            dx = x_coords[i+1] - x_coords[i]
            dy = y_coords[i+1] - y_coords[i]
            psi[i] = np.arctan2(dy, dx)
        psi[-1] = psi[0]
        self.raceline_psi = psi
        
        y_coords_updated = y_coords + [y_coords[0]]
        x_coords_updated = x_coords + [x_coords[0]]
        self.x_spline = CubicSpline(s_coords, x_coords_updated, bc_type='periodic')
        self.y_spline = CubicSpline(s_coords, y_coords_updated, bc_type='periodic')
        self.psi_spline = CubicSpline(s_coords, psi, bc_type='periodic')
        
    def cartesian_to_frenet(self, x, y):
        if self.raceline_s is None:
            raise ValueError("Raceline not initialized")
            
        distances = np.sqrt((self.raceline_x - x)**2 + (self.raceline_y - y)**2)
        k_bar = np.argmin(distances)
        
        dx = x - self.raceline_x[k_bar]
        dy = y - self.raceline_y[k_bar]
        psi_k = self.raceline_psi[k_bar]
        
        s = self.raceline_s[k_bar] + dx * np.cos(psi_k) + dy * np.sin(psi_k)
        d = -dx * np.sin(psi_k) + dy * np.cos(psi_k)
        
        return s, d
        
    def frenet_to_cartesian(self, s, d):
        if self.raceline_s is None:
            raise ValueError("Raceline not initialized")
            
        s_wrapped = s % self.raceline_s[-1]
        
        x_ref = self.x_spline(s_wrapped)
        y_ref = self.y_spline(s_wrapped)
        psi_ref = self.psi_spline(s_wrapped)
        
        x = x_ref - d * np.sin(psi_ref)
        y = y_ref + d * np.cos(psi_ref)
        
        return x, y
        
    def frenet_velocity_to_cartesian(self, s, d, vs, vd):
        if self.raceline_s is None:
            raise ValueError("Raceline not initialized")
            
        s_wrapped = s % self.raceline_s[-1]
        psi_ref = self.psi_spline(s_wrapped)
        
        vx = vs * np.cos(psi_ref) - vd * np.sin(psi_ref)
        vy = vs * np.sin(psi_ref) + vd * np.cos(psi_ref)
        
        return vx, vy
