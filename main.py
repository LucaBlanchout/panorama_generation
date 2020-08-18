import vector
import coordinates
import numpy as np
from PIL import Image
import math
import time
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

viewing_circle_radius = 0.032
optical_centres_radius = 0.15
cameras_angle = (math.pi / 3, math.pi, 5 * math.pi / 3)

panos = []

for i in range(1, 4):
    img = Image.open('images/360render_' + str(i) + '.jpeg')
    img = img.resize((1024, 512), Image.NEAREST)

    panos.append(np.array(img))

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

left_eye_pano = np.zeros((height, width, channels))
right_eye_pano = np.zeros((height, width, channels))
eye_panos = [left_eye_pano, right_eye_pano]
pano_canvas = np.zeros((height, width))

rho = 1
for index, x in np.ndenumerate(pano_canvas):
    col = index[1]
    row = index[0]

    theta, phi = coordinates.spherical_from_latlong(col, row, width, height)

    projection_point.x = (rho * math.cos(theta) * math.sin(phi))
    projection_point.y = (rho * math.sin(theta))
    projection_point.z = - (rho * math.cos(theta) * math.cos(phi))

    for i in range(2):
        try:
            angles = vector.calculate_angles_eye_cameras(eye_vectors[i], camera_vectors)
            sorted_angles = sorted(angles)
            min_angles_index = [angles.index(sorted_angles[0]), angles.index(sorted_angles[1])]

            for j in min_angles_index:
                theta, phi, rho = camera_vectors[j].get_spherical_vector()
                # if row == height / 2:
                #     print(theta, phi)
                u, v = coordinates.latlong_from_spherical(phi, theta, width, height)

                eye_panos[i][row, col] = panos[j][int(v), int(u)]
        except (ValueError, IndexError) as e:
            pass


left_pano = Image.fromarray(eye_panos[0].astype(np.uint8))
left_pano.save("out/left_eye_pano.jpeg")

right_pano = Image.fromarray(eye_panos[1].astype(np.uint8))
right_pano.save("out/right_eye_pano.jpeg")
