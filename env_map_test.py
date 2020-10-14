import utils
import numpy as np
from PIL import Image
import math
from pathlib import Path
import time
from skylibs.envmap import EnvironmentMap
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

viewing_circle_radius = 0.032
optical_centres_radius = 0.15

width_resolution = 2048
number_of_cameras = 3
cameras_to_keep = 2
cameras_to_keep_range = tuple(range(1, cameras_to_keep + 1))
sigma = 3
# type = 'latlong'
type = 'cube'

in_path = 'images/' + str(width_resolution) + "/" + str(number_of_cameras) + '/360render_'
out_path = "out/envmap/" + type + '/' + str(width_resolution) + "/" + str(number_of_cameras) + "/keep_" + str(cameras_to_keep_range[-1]) + "/"
Path(out_path).mkdir(parents=True, exist_ok=True)

camera_angles = []

env_map_panos = []

for i in range(number_of_cameras):
    e = EnvironmentMap(in_path + str(i) + '.jpg', 'latlong')
    if type == 'cube':
        e = e.convertTo('cube')
    env_map_panos.append(e)

    cubemap = Image.fromarray((e.data * 255).astype(np.uint8))
    save_path = out_path + "original_" + str(i) + '.jpg'
    cubemap.save(save_path)

    camera_angles.append(2 * math.pi * i / number_of_cameras)

height, width, channels = env_map_panos[0].data.shape

pano_canvas = utils.create_new_pano_canvas(height, width)
# rho_range = np.linspace(0.5, 5, 10)
rho_range = [3]

for rho in rho_range:
    print("Starting rho =", rho)
    panos_side = ['_left_eye.jpg', '_right_eye.jpg']

    x, y, z, valid = env_map_panos[0].worldCoordinates()

    valid_2d = np.stack((valid, valid)).transpose((1, 2, 0))
    valid_2d = valid_2d.reshape(-1, cameras_to_keep)

    valid_3d = np.stack((valid, valid, valid)).transpose((1, 2, 0))
    valid_3d = valid_3d.reshape(-1, 3)

    new_panos_thetas, new_panos_phis = utils.spherical_from_latlong_envmap_2(type, x, y, z)

    projection_points = utils.create_all_projection_points(rho, new_panos_thetas, new_panos_phis)

    eye_points = utils.create_all_eyes_points(projection_points)

    cameras_vectors_cartesian, cameras_vectors_spherical = utils.create_all_cameras_vectors(projection_points,
                                                                                            camera_angles,
                                                                                            optical_centres_radius)

    new_panos = []
    min_angles_indexes = []
    angles_weights = []
    uvs = []

    for new_pano_index in range(2):
        new_panos.append(np.zeros((height, width, channels)).reshape((-1, 3)))

        eye_vectors = utils.create_eye_vectors(projection_points, eye_points[new_pano_index])

        angles = []

        for camera_vectors in cameras_vectors_cartesian:
            angles.append(utils.calculate_angles_between_vectors(eye_vectors, camera_vectors))

        angles = np.transpose(np.array(angles))

        min_angles_index = utils.calculate_minimum_angles_and_indexes(angles, cameras_to_keep_range)[1]

        min_angles_indexes.append(min_angles_index)

        best_cameras_vectors_spherical = cameras_vectors_spherical[
                                         min_angles_index,
                                         np.arange(cameras_vectors_spherical.shape[1])[:, None],
                                         :
                                         ]

        print(best_cameras_vectors_spherical.shape)

        xyz = utils.calculate_xyz_position_from_camera_vectors_envmap(best_cameras_vectors_spherical, valid_3d)

        print(xyz.shape)

        uv = utils.calculate_latlong_position_from_camera_vectors_envmap(
            best_cameras_vectors_spherical,
            width,
            height,
            valid_2d
        )

        print(uv.shape)

        uvs.append(uv)

    for pano_side, new_pano, min_angles_index, uv in zip(panos_side, new_panos, min_angles_indexes, uvs):
        for i in range(len(env_map_panos)):
            for j in range(cameras_to_keep):
                index = np.argwhere(min_angles_index[:, j] == i)

                new_pano[index, :] += env_map_panos[i].data[uv[index, j, 1], uv[index, j, 0], :]

        if type == 'cube':
            pano_lat_long = EnvironmentMap(new_pano.reshape(height, width, channels), 'cube')
            pano_lat_long = pano_lat_long.convertTo('latlong')
            eye_pano = Image.fromarray((pano_lat_long.data * 255 / 2).astype(np.uint8))
        else:
            eye_pano = Image.fromarray((new_pano.reshape(height, width, channels) * 255 / 2).astype(np.uint8))

        save_path = out_path + "rho_" + str(rho) + "_with_mask" + pano_side
        eye_pano.save(save_path)
        print("Saved in :", save_path)

    print("Finished rho =", rho, "\n")



timer = time.time()
for rho in rho_range:
    print("Starting rho =", rho)
    panos_side = ['_left_eye.jpg', '_right_eye.jpg']

    # im_coor = np.array(env_map_panos[0].imageCoordinates()).transpose((1, 2, 0))
    # mask = np.array(env_map_panos[0].worldCoordinates()[3])
    #
    # mask = np.stack((mask, mask)).transpose((1, 2, 0))
    #
    # im_coor_masked = np.where(mask == 1.0, im_coor, np.nan)
    # new_panos_thetas, new_panos_phis = utils.spherical_from_latlong_envmap(im_coor_masked)

    pano_canvas = utils.create_new_pano_canvas(height, width)

    new_panos_thetas, new_panos_phis = utils.spherical_from_latlong(pano_canvas, width, height)

    projection_points = utils.create_all_projection_points(rho, new_panos_thetas, new_panos_phis)

    eye_points = utils.create_all_eyes_points(projection_points)

    cameras_vectors_cartesian, cameras_vectors_spherical = utils.create_all_cameras_vectors(projection_points,
                                                                                            camera_angles,
                                                                                            optical_centres_radius)

    new_panos = []
    min_angles_indexes = []
    angles_weights = []
    uvs = []

    for new_pano_index in range(2):
        new_panos.append(np.zeros((height, width, channels)).reshape((-1, 3)))

        eye_vectors = utils.create_eye_vectors(projection_points, eye_points[new_pano_index])

        angles = []

        for camera_vectors in cameras_vectors_cartesian:
            angles.append(utils.calculate_angles_between_vectors(eye_vectors, camera_vectors))

        angles = np.transpose(np.array(angles))

        min_angles, min_angles_index = utils.calculate_minimum_angles_and_indexes(angles, cameras_to_keep_range)

        min_angles_indexes.append(min_angles_index)

        angles_weights.append(utils.calculate_weights_of_angles(min_angles, sigma))

        best_cameras_vectors_spherical = cameras_vectors_spherical[
            min_angles_index,
            np.arange(cameras_vectors_spherical.shape[1])[:, None],
            :
        ]

        uvs.append(utils.calculate_latlong_position_from_camera_vectors(
            best_cameras_vectors_spherical,
            width,
            height
        ))

    for pano_side, new_pano, min_angles_index, angles_weight, uv in zip(panos_side, new_panos, min_angles_indexes,
                                                                        angles_weights, uvs):
        for i in range(len(env_map_panos)):
            for j in range(cameras_to_keep):
                index = np.argwhere(min_angles_index[:, j] == i)

                # new_pano[index, :] += env_map_panos[i].data[uv[index, j, 1], uv[index, j, 0], :]
                new_pano[index, :] += angles_weight[index, j][:, None] * env_map_panos[i].data[uv[index, j, 1], uv[index, j, 0], :]

        if type == 'cube':
            pano_lat_long = EnvironmentMap(new_pano.reshape(height, width, channels), 'cube')
            pano_lat_long = pano_lat_long.convertTo('latlong')
            eye_pano = Image.fromarray((pano_lat_long.data * 255).astype(np.uint8))
        else:
            eye_pano = Image.fromarray((new_pano.reshape(height, width, channels) * 255).astype(np.uint8))

        save_path = out_path + "rho_" + str(rho) + pano_side
        eye_pano.save(save_path)
        print("Saved in :", save_path)

    print("Finished rho =", rho, "\n")

print("\nTime=", time.time() - timer)
