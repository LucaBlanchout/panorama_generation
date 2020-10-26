import numpy as np


class Camera:
    def __init__(self, angle, optical_centres_radius=0.15):
        self.angle = angle
        self.optical_centres_radius = 0.15

        self.coordinates = np.array([
            optical_centres_radius * np.cos(self.angle),
            0.,
            -optical_centres_radius * np.sin(self.angle)
        ])

        self.vectors_cartesian = None
        self.vectors_spherical = None

    def calculate_vectors(self, projection_points):
        vx = projection_points[:, 0] - self.coordinates[0]
        vy = projection_points[:, 1] - self.coordinates[1]
        vz = projection_points[:, 2] - self.coordinates[2]

        magnitude = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

        vx /= magnitude
        vy /= magnitude
        vz /= magnitude

        thetas = np.arcsin(vy)
        phis = np.arctan2(vx, -1 * vz)

        self.vectors_cartesian = np.stack((vx, vy, vz), axis=1)
        self.vectors_spherical = np.stack((thetas, phis), axis=1)


class CameraContainer:
    def __init__(self, cameras=None):
        self.cameras = cameras if cameras is not None else []

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, item):
        return self.cameras[item]

    def __iter__(self):
        yield from self.cameras

    def append(self, camera):
        self.cameras.append(camera)

    def get_cameras_coordinates(self):
        cameras_coordinates = []
        for camera in self:
            cameras_coordinates.append(camera.coordinates)

        return np.array(cameras_coordinates)

    def calculate_cameras_vectors(self, projection_points):
        for camera in self:
            camera.calculate_vectors(projection_points)

    def get_cameras_vectors_cartesian(self):
        camera_vectors_cartesian = []
        for camera in self:
            camera_vectors_cartesian.append(camera.vectors_cartesian)

        return np.asarray(camera_vectors_cartesian)
