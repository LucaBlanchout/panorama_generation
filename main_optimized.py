import utils
import numpy as np
from PIL import Image
import math
from pathlib import Path
import time
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

viewing_circle_radius = 0.032
optical_centres_radius = 0.15

width_resolution = 2048
number_of_cameras = 3
cameras_to_keep = 2
cameras_to_keep = tuple(range(1, cameras_to_keep + 1))
sigma = 1

path = "out/cameras_" + str(number_of_cameras) + "/keep_" + str(cameras_to_keep[-1]) + "/" + str(width_resolution) + "/"

Path(path).mkdir(parents=True, exist_ok=True)

camera_angles = []
panos = []

for i in range(number_of_cameras):
    img = Image.open('images/' + str(width_resolution) + "/" + str(number_of_cameras) + '/360render_' + str(i) + '.jpg')
    panos.append(np.array(img))

    camera_angles.append(2 * math.pi * i / number_of_cameras)

# print(panos.shape)
# channels = 3
# height = 4
# width = 8
height, width, channels = panos[0].shape
rho_range = [3]
# rho_range = np.linspace(0.5, 5, 10)

timer = time.time()
for rho in rho_range:
    left_eye_pano = np.zeros((height, width, channels)).reshape((-1, 3))
    right_eye_pano = np.zeros((height, width, channels)).reshape((-1, 3))
    eye_panos = [left_eye_pano, right_eye_pano]
    panos_side = ['_left_eye.jpg', '_right_eye.jpg']
    pano_canvas = utils.create_canvas(height, width)

    thetas, phis = utils.spherical_from_latlong_vectorized(pano_canvas, width, height)

    projection_points = utils.create_all_projection_points(rho, thetas, phis)

    left_eye_vectors, right_eye_vectors = utils.create_all_eyes_vectors(projection_points)

    cameras_vectors_cartesian, cameras_vectors_spherical = utils.create_all_cameras_vectors(projection_points,
                                                                                            camera_angles,
                                                                                            optical_centres_radius)

    left_eye_angles = []
    right_eye_angles = []

    for camera_vectors in cameras_vectors_cartesian:
        left_eye_angles.append(utils.calculate_angles_between_vectors(left_eye_vectors, camera_vectors))
        right_eye_angles.append(utils.calculate_angles_between_vectors(right_eye_vectors, camera_vectors))

    left_eye_angles = np.transpose(np.asarray(left_eye_angles))
    right_eye_angles = np.transpose(np.asarray(right_eye_angles))

    min_angles_left_eye, min_angles_indexes_left_eye = utils.calculate_minimum_angles_and_indexes(left_eye_angles, cameras_to_keep)
    min_angles_right_eye, min_angles_indexes_right_eye = utils.calculate_minimum_angles_and_indexes(right_eye_angles, cameras_to_keep)

    angles_weights_left_eye = utils.calculate_weights_of_angles(min_angles_left_eye, sigma)
    angles_weights_right_eye = utils.calculate_weights_of_angles(min_angles_right_eye, sigma)

    angles_weights = [angles_weights_left_eye, angles_weights_right_eye]

    min_angles_indexes = [min_angles_indexes_left_eye, min_angles_indexes_right_eye]

    best_cameras_vectors_spherical_left_eye = cameras_vectors_spherical[
                                              min_angles_indexes_left_eye,
                                              np.arange(cameras_vectors_spherical.shape[1])[:, None],
                                              :
                                              ]

    best_cameras_vectors_spherical_right_eye = cameras_vectors_spherical[
                                               min_angles_indexes_right_eye,
                                               np.arange(cameras_vectors_spherical.shape[1])[:, None],
                                               :
                                               ]

    left_uvs = utils.calculate_latlong_position_from_camera_vectors(best_cameras_vectors_spherical_left_eye, width,
                                                                    height)
    right_uvs = utils.calculate_latlong_position_from_camera_vectors(best_cameras_vectors_spherical_right_eye, width,
                                                                     height)

    uvs = [left_uvs, right_uvs]

    for pano_side, eye_pano, min_angles_index, angles_weight, uv in zip(panos_side, eye_panos, min_angles_indexes, angles_weights, uvs):
        for i in range(len(panos)):
            for j in range(cameras_to_keep[-1]):
                index = np.argwhere(min_angles_index[:, j] == i)

                eye_pano[index, :] += (angles_weight[index, j][:, None] * panos[i][uv[index, j, 1], uv[index, j, 0], :]).astype(np.uint8)

        eye_pano = Image.fromarray(eye_pano.reshape(height, width, channels).astype(np.uint8))
        save_path = path + "rho_" + str(rho) + pano_side
        eye_pano.save(save_path)
        print("Saved in :", save_path)

    # for pano_side, eye_pano, min_angles_index, angles_weight, uv in zip(panos_side, eye_panos, min_angles_indexes, angles_weights, uvs):
    #     it = np.nditer(min_angles_index, flags=['multi_index'])
    #
    #     for min_angle_index in it:
    #         i, j = it.multi_index
    #         try:
    #             eye_pano[i, :] += angles_weight[i, j] * panos[min_angle_index][uv[i, j, 1], uv[i, j, 0], :]
    #         except IndexError:
    #             pass
    #
    #     # eye_pano /= cameras_to_keep[-1]
    #     eye_pano = Image.fromarray(eye_pano.reshape(height, width, channels).astype(np.uint8))
    #     save_path = path + "rho_" + str(rho) + pano_side
    #     eye_pano.save(save_path)
    #     print("Saved in :", save_path)


print("\nTime=", time.time() - timer)
