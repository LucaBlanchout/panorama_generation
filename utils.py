import math
import numpy as np

VIEWING_CIRCLE_RADIUS = 0.032


def old_spherical_from_latlong(u, v, width, height):
    phi = math.pi * (2 * u / width - 1)
    theta = math.pi * (-v / height + 0.5)

    return theta, phi


def old_latlong_from_spherical(phi, theta, width, height):
    u = (width / 2) * ((phi / math.pi) + 1)
    v = height * (0.5 - (theta / math.pi))

    return u, v


def spherical_from_latlong(arr, width, height):
    u = arr[:, :, 1]
    v = arr[:, :, 0]

    phis = math.pi * (2 * u / width - 1)
    thetas = math.pi * (-v / height + 0.5)

    return thetas, phis


def create_new_pano_canvas(m, n):
    return np.indices((m, n)).transpose(1, 2, 0)


def create_all_projection_points(rho, thetas, phis):
    xs = (rho * np.cos(thetas) * np.sin(phis))
    ys = (rho * np.sin(thetas))
    zs = (-rho * np.cos(thetas) * np.cos(phis))

    projection_points = np.stack((xs, ys, zs), axis=2).reshape(-1, 3)

    return projection_points


def create_all_eyes_points(projection_points):
    b = np.sqrt(projection_points[:, 0] ** 2 + projection_points[:, 2] ** 2)

    viewing_circle_radius_array = np.empty(b.shape)
    viewing_circle_radius_array.fill(VIEWING_CIRCLE_RADIUS)

    th = np.arccos(np.divide(viewing_circle_radius_array, b))

    d = np.arctan2(projection_points[:, 2], projection_points[:, 0])

    d1 = d - th
    d2 = d + th

    xs_left = VIEWING_CIRCLE_RADIUS * np.cos(d1)
    zs_left = VIEWING_CIRCLE_RADIUS * np.sin(d1)

    xs_right = VIEWING_CIRCLE_RADIUS * np.cos(d2)
    zs_right = VIEWING_CIRCLE_RADIUS * np.sin(d2)

    left_eye_points = np.stack((xs_left, np.zeros(xs_left.shape), zs_left), axis=1)
    right_eye_points = np.stack((xs_right, np.zeros(xs_right.shape), zs_right), axis=1)

    left_eye_points = np.nan_to_num(left_eye_points)
    right_eye_points = np.nan_to_num(right_eye_points)

    return left_eye_points, right_eye_points


def create_eye_vectors(projection_points, eye_point):
    # eye_points = create_all_eyes_points(projection_points)

    # eye_vectors = []
    #
    # for eye_point in eye_points:
    vx = projection_points[:, 0] - eye_point[:, 0]
    vy = projection_points[:, 1] - eye_point[:, 1]
    vz = projection_points[:, 2] - eye_point[:, 2]

    magnitude = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    vx /= magnitude
    vy /= magnitude
    vz /= magnitude

    eye_vector = np.stack((vx, vy, vz), axis=1)

    # eye_vectors.append(eye_vector)

    return eye_vector


def create_all_cameras_vectors(projection_points, camera_angles, optical_centres_radius):
    cameras_vectors_cartesian = []
    cameras_vectors_spherical = []

    for camera_angle in camera_angles:
        camera_point_x = - optical_centres_radius * np.sin(camera_angle)
        camera_point_z = - optical_centres_radius * np.cos(camera_angle)

        vx = projection_points[:, 0] - camera_point_x
        vy = projection_points[:, 1] - 0.
        vz = projection_points[:, 2] - camera_point_z

        magnitude = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

        vx /= magnitude
        vy /= magnitude
        vz /= magnitude

        thetas = np.arcsin(vy)
        phis = np.arctan2(vx, -1 * vz)

        camera_vectors_cartesian = np.stack((vx, vy, vz), axis=1)
        camera_vectors_spherical = np.stack((thetas, phis), axis=1)

        cameras_vectors_cartesian.append(camera_vectors_cartesian)
        cameras_vectors_spherical.append(camera_vectors_spherical)

    return np.asarray(cameras_vectors_cartesian), np.asarray(cameras_vectors_spherical)


def calculate_angles_between_vectors(v1, v2):
    def dot_product(vector1, vector2):
        return np.sum((vector1 * vector2), axis=1)

    def length(v):
        return np.sqrt(dot_product(v, v))

    ang_rad = np.arccos(dot_product(v1, v2) / length(v1) * length(v2))

    ang_deg = np.degrees(ang_rad) % 360

    np.where(ang_deg - 180 >= 0, 360 - ang_deg, ang_deg)

    return ang_deg


def calculate_minimum_angles_and_indexes(eye_angles, cameras_to_keep):
    min_angles_indexes = np.argpartition(eye_angles, cameras_to_keep)[:, :cameras_to_keep[-1]]
    min_angles = eye_angles[np.arange(eye_angles.shape[0])[:, None], min_angles_indexes]

    return min_angles, min_angles_indexes


def calculate_weights_of_angles(angles, sigma):
    wis = np.exp(-angles / (sigma ** 2))
    wjs = np.sum(wis, axis=1).transpose()
    wis = wis / wjs[:, None]

    return wis


def calculate_latlong_position_from_camera_vectors(cameras_vectors_spherical, width, height):
    uvs = []

    for i in range(cameras_vectors_spherical.shape[1]):
        thetas = cameras_vectors_spherical[:, i, 0]
        phis = cameras_vectors_spherical[:, i, 1]

        us = (width / 2) * ((phis / math.pi) + 1)
        vs = height * (0.5 - (thetas / math.pi))

        us = np.where(us >= width, width - 1, us)
        vs = np.where(vs >= height, height - 1, vs)

        uvs.append(np.stack((us, vs), axis=1))

    uvs = np.array(uvs).transpose((1, 0, 2)).astype(np.int)

    return uvs
