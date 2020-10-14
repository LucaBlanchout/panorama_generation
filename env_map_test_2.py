import utils_envmap
import numpy as np
from PIL import Image
import math
from pathlib import Path
import time
from skylibs.envmap import EnvironmentMap
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})


width_resolution = 2048
number_of_cameras = 3
cameras_to_keep = 2
cameras_to_keep_range = tuple(range(1, cameras_to_keep + 1))
# envmap_type = 'latlong'
envmap_type = 'cube'

in_path = 'images/' + str(width_resolution) + "/" + str(number_of_cameras) + '/360render_'
out_path = "out/envmap/" + envmap_type + '/' + str(width_resolution) + "/" + str(number_of_cameras) + "/keep_" + str(cameras_to_keep_range[-1]) + "/"
Path(out_path).mkdir(parents=True, exist_ok=True)

camera_angles = []

env_map_panos = []

for i in range(number_of_cameras):
    e = EnvironmentMap(in_path + str(i) + '.jpg', 'latlong')
    if envmap_type == 'cube':
        e = e.convertTo('cube')
    env_map_panos.append(e)

    cubemap = Image.fromarray((e.data * 255).astype(np.uint8))

    save_path = out_path + "original_" + str(i) + '.jpg'
    cubemap.save(save_path)

    camera_angles.append(2 * math.pi * i / number_of_cameras)

height, width, channels = env_map_panos[0].data.shape

rho_range = [3]

for rho in rho_range:
    print("Starting rho =", rho)
    panos_side = ['_left_eye.jpg', '_right_eye.jpg']

    x, y, z, valid = env_map_panos[0].worldCoordinates()

    projection_points = utils_envmap.create_all_projection_points(rho, x, y, z)

    eye_points = utils_envmap.create_all_eyes_points(projection_points)

    cameras_vectors_cartesian, cameras_vectors_spherical = utils_envmap.create_all_cameras_vectors(projection_points, camera_angles)

    new_panos = []
    min_angles_indexes = []
    angles_weights = []
    uvs = []
    bests_cameras_vectors_cartesian = []

    for new_pano_index in range(2):
        new_panos.append(np.zeros((height, width, channels)).reshape((-1, 3)))

        eye_vectors = utils_envmap.create_eye_vectors(projection_points, eye_points[new_pano_index])

        angles = []

        for camera_vectors in cameras_vectors_cartesian:
            angles.append(utils_envmap.calculate_angles_between_vectors(eye_vectors, camera_vectors))

        angles = np.transpose(np.array(angles))

        min_angles_index = utils_envmap.calculate_minimum_angles_indexes(angles, cameras_to_keep_range)

        min_angles_indexes.append(min_angles_index)

        best_cameras_vectors_cartesian = cameras_vectors_cartesian[
            min_angles_index,
            np.arange(cameras_vectors_cartesian.shape[1])[:, None],
            :
        ]

        bests_cameras_vectors_cartesian.append(best_cameras_vectors_cartesian)

    for pano_side, new_pano, min_angles_index, best_cameras_vectors_cartesian in zip(panos_side, new_panos, min_angles_indexes, bests_cameras_vectors_cartesian):
        for i in range(len(env_map_panos)):
            for j in range(cameras_to_keep):
                index = np.argwhere(min_angles_index[:, j] == i)

                xyz = np.squeeze(best_cameras_vectors_cartesian[index, j, :])

                u, v = env_map_panos[i].world2image(xyz[:, 0], xyz[:, 1], xyz[:, 2])

                u = (u * width).astype(np.int)
                v = (v * height).astype(np.int)

                new_pano[index, :] += env_map_panos[i].data[v, u, :][:, None]

        pano_lat_long = EnvironmentMap(new_pano.reshape(height, width, channels), 'cube')
        eye_pano = Image.fromarray((new_pano.reshape(height, width, channels) * 255 / 2).astype(np.uint8))
        save_path = out_path + "cubemap_rho_" + str(rho) + pano_side
        eye_pano.save(save_path)

        pano_lat_long = pano_lat_long.convertTo('latlong')
        eye_pano = Image.fromarray((pano_lat_long.data * 255 / 2).astype(np.uint8))

        save_path = out_path + "rho_" + str(rho) + pano_side
        eye_pano.save(save_path)
        print("Saved in :", save_path)







