import vector
import coordinates
import numpy as np
from PIL import Image
import math
from pathlib import Path

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

viewing_circle_radius = 0.032
optical_centres_radius = 0.15

width_resolution = 4096
number_of_cameras = 24
cameras_to_keep = 1

path = "out/cameras_" + str(number_of_cameras) + "/keep_" + str(cameras_to_keep) + "/" + str(width_resolution) + "/"

Path(path).mkdir(parents=True, exist_ok=True)

cameras_angle = []
panos = []

for i in range(number_of_cameras):
    img = Image.open('images/' + str(width_resolution) + "/" + str(number_of_cameras) + '/360render_' + str(i) + '.jpg')
    panos.append(np.array(img))

    cameras_angle.append(2 * math.pi * i / number_of_cameras)

projection_point = vector.Point(0., 0., -1.)

left_eye_point = vector.EyePoint(projection_point, side='left')
right_eye_point = vector.EyePoint(projection_point, side='right')

left_eye_vector = vector.Vector(projection_point, left_eye_point)
right_eye_vector = vector.Vector(projection_point, right_eye_point)

eye_vectors = [left_eye_vector, right_eye_vector]

camera_vectors = []

for camera_angle in cameras_angle:
    camera_point = vector.Point()

    camera_point.x = - optical_centres_radius * math.sin(camera_angle)
    camera_point.z = - optical_centres_radius * math.cos(camera_angle)

    camera_vectors.append(vector.Vector(projection_point, camera_point))

height, width, channels = panos[0].shape

# rho_range = np.linspace(0.5, 5, 10)
rho_range = [4]
for rho in rho_range:
    print("Starting rho =", rho)
    left_eye_pano = np.zeros((height, width, channels))
    right_eye_pano = np.zeros((height, width, channels))
    eye_panos = [left_eye_pano, right_eye_pano]
    pano_canvas = np.zeros((height, width))

    for index, x in np.ndenumerate(pano_canvas):
        col = index[1]
        row = index[0]

        theta, phi = coordinates.spherical_from_latlong(col, row, width, height)

        projection_point.x = (rho * math.cos(theta) * math.sin(phi))
        projection_point.y = (rho * math.sin(theta))
        projection_point.z = - (rho * math.cos(theta) * math.cos(phi))

        for i in range(len(eye_vectors)):
            try:
                angles = vector.calculate_angles_eye_cameras(eye_vectors[i], camera_vectors)
                sorted_angles = sorted(angles)
                min_angles_index = [angles.index(sorted_angles[k]) for k in range(cameras_to_keep)]

                for j in min_angles_index:
                    theta, phi = camera_vectors[j].get_spherical_vector()
                    u, v = coordinates.latlong_from_spherical(phi, theta, width, height)

                    eye_panos[i][row, col] += panos[j][int(v), int(u)]
            except (ValueError, IndexError) as e:
                pass

    eye_panos[0] = eye_panos[0] // cameras_to_keep
    eye_panos[1] = eye_panos[1] // cameras_to_keep

    left_pano = Image.fromarray(eye_panos[0].astype(np.uint8))
    left_pano.save(path + "rho_" + str(rho) + "_left_eye.jpg")

    right_pano = Image.fromarray(eye_panos[1].astype(np.uint8))
    right_pano.save(path + "rho_" + str(rho) + "_right_eye.jpg")

    print("Rho =", str(rho), "is done")
