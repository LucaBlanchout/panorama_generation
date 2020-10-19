import utils_envmap
import numpy as np
from PIL import Image
import math
from pathlib import Path
from skylibs.envmap import EnvironmentMap
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})


def plot_image(img):
    plt.imshow(img)
    plt.show()


width_resolution = 2048
number_of_cameras = 3
cameras_to_keep = 2
cameras_to_keep_range = tuple(range(1, cameras_to_keep + 1))
# envmap_type = 'latlong'
envmap_type = 'cube'

in_path = 'images/' + str(width_resolution) + "/" + str(number_of_cameras) + '/360render_'
out_path = "out/envmap/" + envmap_type + '/' + str(width_resolution) + "/" + str(number_of_cameras) + "/keep_" + str(
    cameras_to_keep_range[-1]) + "/"
Path(out_path).mkdir(parents=True, exist_ok=True)

camera_angles = []

env_map_panos = []

for i in range(number_of_cameras):
    e = EnvironmentMap(in_path + str(i) + '.jpg', 'latlong')
    if envmap_type == 'cube':
        e = e.convertTo('cube')
    env_map_panos.append(e)

    angle = (1 + 2 * i) * math.pi / number_of_cameras + math.pi / 2
    camera_angles.append(angle)

height, width, channels = env_map_panos[0].data.shape

rho_range = np.linspace(0.5, 5, 10)

for rho in rho_range:
    print("Starting rho =", rho)
    pano_side_list = ['_left_eye.jpg', '_right_eye.jpg']

    camera_points = utils_envmap.create_all_camera_points(camera_angles)

    x, y, z, valid = env_map_panos[0].worldCoordinates()

    projection_points = utils_envmap.create_all_projection_points(rho, x, y, z)

    eye_points = utils_envmap.create_all_eyes_points(projection_points)

    camera_vectors_cartesian_coordinates, camera_vectors_spherical_coordinates = utils_envmap.create_all_cameras_vectors(
        projection_points, camera_angles)

    new_pano_list = []
    min_angles_index_list = []
    min_angles_ratio_list = []
    best_cameras_vectors_cartesian_list = []

    for new_pano_index in range(2):
        new_pano_list.append(np.zeros((height, width, channels)).reshape((-1, 3)))

        eye_vectors = utils_envmap.create_eye_vectors(projection_points, eye_points[new_pano_index])

        angles = []

        for camera_vectors in camera_vectors_cartesian_coordinates:
            angles.append(utils_envmap.calculate_angles_between_vectors(eye_vectors, camera_vectors))

        angles = np.transpose(np.array(angles))

        min_angles, min_angles_index = utils_envmap.calculate_minimum_angles_and_indexes(angles, cameras_to_keep_range)

        min_angles_ratio = utils_envmap.calculate_angles_ratio(min_angles)

        intermediate_points = utils_envmap.calculate_intermediate_points(camera_points, min_angles_ratio, min_angles_index)

        min_angles_index_list.append(min_angles_index)

        min_angles_ratio_list.append(min_angles_ratio)

        best_camera_vectors_cartesian_coordinates = camera_vectors_cartesian_coordinates[
                                                    min_angles_index,
                                                    np.arange(camera_vectors_cartesian_coordinates.shape[1])[:, None],
                                                    :
                                                    ]

        best_cameras_vectors_cartesian_list.append(best_camera_vectors_cartesian_coordinates)

    for pano_side, new_pano, min_angles_index, best_camera_vectors_cartesian_coordinates in zip(pano_side_list,
                                                                                                new_pano_list,
                                                                                                min_angles_index_list,
                                                                                                best_cameras_vectors_cartesian_list):
        for i in range(len(env_map_panos)):
            for j in range(cameras_to_keep):
                index = np.argwhere(min_angles_index[:, j] == i)

                xyz = np.squeeze(best_camera_vectors_cartesian_coordinates[index, j, :])

                u, v = env_map_panos[i].world2image(xyz[:, 0], xyz[:, 1], xyz[:, 2])

                u = (u * width).astype(np.int)
                v = (v * height).astype(np.int)

                new_pano[index, :] += env_map_panos[i].data[v, u, :][:, None]

        new_pano = (new_pano.reshape(height, width, channels) * 255 / cameras_to_keep).astype(np.uint8)

        if envmap_type == 'cube':
            eye_pano_cube = Image.fromarray(new_pano)
            save_path = out_path + "cubemap_rho_" + str(rho) + pano_side
            eye_pano_cube.save(save_path)
            print("Saved cubemap representation in :", save_path)

            eye_pano_latlong = EnvironmentMap(new_pano, 'cube')
            eye_pano_latlong = eye_pano_latlong.convertTo('latlong')
            eye_pano_latlong = Image.fromarray(eye_pano_latlong.data.astype(np.uint8))
            save_path = out_path + "rho_" + str(rho) + pano_side
            eye_pano_latlong.save(save_path)
        else:
            eye_pano_latlong = Image.fromarray(new_pano)
            save_path = out_path + "rho_" + str(rho) + pano_side
            eye_pano_latlong.save(save_path)

        print("Saved latlong representation in :", save_path)
