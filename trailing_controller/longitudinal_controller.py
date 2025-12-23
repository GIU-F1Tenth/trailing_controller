#!/usr/bin/env python3

import numpy as np


class LongitudinalController:
    def __init__(self, kp=1.0, kd=0.2, target_gap=2.0, v_blind=1.5, track_length=100.0):
        self.kp = kp
        self.kd = kd
        self.target_gap = target_gap
        self.v_blind = v_blind
        self.track_length = track_length
        
    def compute_velocity(self, ego_s, ego_vs, opp_s, opp_vs):
        delta_s = (ego_s - opp_s) % self.track_length
        
        e_gap = self.target_gap - delta_s
        
        delta_vs = ego_vs - opp_vs
        
        v_des = opp_vs - (self.kp * e_gap + self.kd * delta_vs)
        
        v_des_final = max(self.v_blind, v_des)
        
        return v_des_final
