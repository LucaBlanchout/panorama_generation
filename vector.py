from math import sqrt, sin, cos, acos, atan2, degrees


class Vector:
    def __init__(self, projection_point, other_point):
        self.projection_point = projection_point
        self.other_point = other_point
        self._vector = None

    def update(self):
        self.other_point.update()
        self._vector = [self.projection_point.x - self.other_point.x,
                        self.projection_point.y - self.other_point.y,
                        self.projection_point.z - self.other_point.z]

    def get_vector(self):
        self.update()
        return self._vector


class Point:
    def __init__(self, x=0., y=0., z=0.):
        self.x = x
        self.y = y
        self.z = z

    def update(self):
        pass


class EyePoint:
    def __init__(self, projection_point, side='left'):
        self.viewing_circle_radius = 0.032
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.side = side
        self.projection_point = projection_point
        self.update()

    def update(self):
        b = sqrt(self.projection_point.x ** 2 + self.projection_point.y ** 2)
        th = acos(self.viewing_circle_radius / b)
        d = atan2(self.projection_point.y, self.projection_point.x)
        if self.side == 'left':
            d = d + th
        else:
            d = d - th

        self.x = self.viewing_circle_radius * cos(d)
        self.y = self.viewing_circle_radius * sin(d)


def calculate_angle_between_vectors(v1, v2):
    def dot_product(vector1, vector2):
        return sum((a * b) for a, b in zip(vector1, vector2))

    def length(v):
        return sqrt(dot_product(v, v))

    v1 = v1.get_vector()
    v2 = v2.get_vector()

    ang_rad = acos(dot_product(v1, v2) / (length(v1) * length(v2)))

    ang_deg = degrees(ang_rad) % 360

    if ang_deg - 180 >= 0:
        return 360 - ang_deg
    else:
        return ang_deg
