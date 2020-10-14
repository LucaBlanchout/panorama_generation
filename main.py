import utils
import numpy as np
from PIL import Image
import math
from pathlib import Path
import time

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

viewing_circle_radius = 0.032
optical_centres_radius = 0.15

width_resolution = 2048
number_of_cameras = 3
cameras_to_keep = 2
cameras_to_keep = tuple(range(1, cameras_to_keep + 1))
sigma = 1

in_path = 'images/' + str(width_resolution) + "/" + str(number_of_cameras) + '/360render_'
out_path = "out/" + str(width_resolution) + "/" + str(number_of_cameras) + "/keep_" + str(cameras_to_keep[-1]) + "/"
Path(out_path).mkdir(parents=True, exist_ok=True)

camera_angles = []
base_panos = []

for i in range(number_of_cameras):
    img = Image.open(in_path + str(i) + '.jpg')
    base_panos.append(np.array(img))

    camera_angles.append(2 * math.pi * i / number_of_cameras)

base_panos = np.array(base_panos)
height, width, channels = base_panos[0].shape
rho_range = np.linspace(0.5, 5, 10)

timer = time.time()
for rho in rho_range:
    print("Starting rho =", rho)
    panos_side = ['_left_eye.jpg', '_right_eye.jpg']
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

        min_angles, min_angles_index = utils.calculate_minimum_angles_and_indexes(angles, cameras_to_keep)

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
        for i in range(base_panos.shape[0]):
            for j in range(cameras_to_keep[-1]):
                index = np.argwhere(min_angles_index[:, j] == i)

                print(new_pano.shape)
                print(new_pano[index, :].shape, '\n')

                print(base_panos[i][uv[index, j, 1], uv[index, j, 0], :].shape)

                new_pano[index, :] += (angles_weight[index, j][:, None] * base_panos[i][uv[index, j, 1], uv[index, j, 0], :]).astype(np.uint8)

        eye_pano = Image.fromarray(new_pano.reshape(height, width, channels).astype(np.uint8))
        save_path = out_path + "rho_" + str(rho) + pano_side
        eye_pano.save(save_path)
        print("Saved in :", save_path)

    print("Finished rho =", rho)

print("\nTime=", time.time() - timer)
