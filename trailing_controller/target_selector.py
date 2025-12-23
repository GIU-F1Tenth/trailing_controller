import numpy as np
from typing import List, Optional, Tuple


class TargetSelector:
    def __init__(self, 
                 max_lateral_distance: float = 3.0,
                 max_longitudinal_distance: float = 20.0,
                 prefer_ahead: bool = True):
        self.max_lateral_distance = max_lateral_distance
        self.max_longitudinal_distance = max_longitudinal_distance
        self.prefer_ahead = prefer_ahead
        self.current_target_id = None
        self.target_lost_count = 0
        self.max_target_lost = 5
        
    def select_target(self, objects: List) -> Optional[int]:
        if len(objects) == 0:
            if self.current_target_id is not None:
                self.target_lost_count += 1
                if self.target_lost_count > self.max_target_lost:
                    self.current_target_id = None
                    self.target_lost_count = 0
            return None
        
        valid_objects = []
        for obj in objects:
            if obj.is_static:
                continue
                
            lateral_dist = abs(obj.d)
            longitudinal_dist = abs(obj.s)
            
            if lateral_dist > self.max_lateral_distance:
                continue
            if longitudinal_dist > self.max_longitudinal_distance:
                continue
            if self.prefer_ahead and obj.s < 0:
                continue
                
            valid_objects.append(obj)
        
        if len(valid_objects) == 0:
            if self.current_target_id is not None:
                self.target_lost_count += 1
                if self.target_lost_count > self.max_target_lost:
                    self.current_target_id = None
                    self.target_lost_count = 0
            return None
        
        if self.current_target_id is not None:
            for obj in valid_objects:
                if obj.id == self.current_target_id:
                    self.target_lost_count = 0
                    return obj.id
        
        best_obj = min(valid_objects, key=lambda o: np.sqrt(o.s**2 + o.d**2))
        self.current_target_id = best_obj.id
        self.target_lost_count = 0
        
        return best_obj.id
    
    def get_target_object(self, objects: List, target_id: int) -> Optional:
        for obj in objects:
            if obj.id == target_id:
                return obj
        return None
