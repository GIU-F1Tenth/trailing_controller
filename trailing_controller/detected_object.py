class DetectedObject:
    def __init__(self, obj_id, x, y, s=0.0, d=0.0, vs=0.0, vd=0.0, is_static=False, age=0):
        self.id = obj_id
        self.x = x
        self.y = y
        self.s = s
        self.d = d
        self.vs = vs
        self.vd = vd
        self.is_static = is_static
        self.age = age
