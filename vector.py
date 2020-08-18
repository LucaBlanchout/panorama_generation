from math import sqrt, sin, cos, asin, acos, atan, atan2, degrees


class Vector:
    def __init__(self, projection_point, other_point):
        self.projection_point = projection_point
        self.other_point = other_point
        self.vx = None
        self.vy = None
        self.vz = None
        self.magnitude = None
        self._vector = None
        self.update()

    def update(self):
        self.other_point.update()
        self.vx = self.projection_point.x - self.other_point.x
        self.vy = self.projection_point.y - self.other_point.y
        self.vz = self.projection_point.z - self.other_point.z

        self.magnitude = sqrt((self.vx ** 2) + (self.vy ** 2) + (self.vz ** 2))

        self._vector = [self.vx / self.magnitude,
                        self.vy / self.magnitude,
                        self.vz / self.magnitude]

    def get_cartesian_vector(self):
        self.update()
        return self._vector

    def get_spherical_vector(self):
        theta = asin(self.vy)
        phi = atan2(self.vx, -self.vz)

        return theta, phi, 1


class Point:
    def __init__(self, x=0., y=0., z=0.):
        self.x = x
        self.y = y
        self.z = z

    def update(self):
        pass

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.z)


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
        b = sqrt(self.projection_point.x ** 2 + self.projection_point.z ** 2)
        th = acos(self.viewing_circle_radius / b)
        d = atan2(self.projection_point.z, self.projection_point.x)
        if self.side == 'left':
            d = d - th
        else:
            d = d + th

        self.x = self.viewing_circle_radius * cos(d)
        self.z = self.viewing_circle_radius * sin(d)

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.z)


def calculate_angle_between_vectors(v1, v2):
    def dot_product(vector1, vector2):
        return sum((a * b) for a, b in zip(vector1, vector2))

    def length(v):
        return sqrt(dot_product(v, v))

    v1 = v1.get_cartesian_vector()
    v2 = v2.get_cartesian_vector()

    ang_rad = acos(dot_product(v1, v2) / (length(v1) * length(v2)))

    ang_deg = degrees(ang_rad) % 360

    if ang_deg - 180 >= 0:
        return 360 - ang_deg
    else:
        return ang_deg


def calculate_angles_eye_cameras(eye_vector, camera_vectors):
    angles = []
    for camera_vector in camera_vectors:
        angle = calculate_angle_between_vectors(eye_vector, camera_vector)
        angles.append(angle)

    return angles
