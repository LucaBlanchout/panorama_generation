import vector
import numpy as np
from PIL import Image
import math

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

camera_vectors = []

for camera_angle in cameras_angle:
    camera_point = vector.Point()

    camera_point.x = - optical_centres_radius * math.sin(camera_angle)
    camera_point.z = - optical_centres_radius * math.cos(camera_angle)

    camera_vectors.append(vector.Vector(projection_point, camera_point))


print(left_eye_vector.get_vector())
print(right_eye_vector.get_vector())
for camera_vector in camera_vectors:
    print(camera_vector.get_vector())


height, width, channels = panos[0].shape

left_eye_pano = np.zeros((height, width, channels))
left_eye_pano_canvas = np.zeros((height, width))

rho = 2
for index, x in np.ndenumerate(left_eye_pano_canvas):
    col = index[1]
    row = index[0]

    phi = math.pi * (col / height - 1)
    theta = math.pi * (-row / height + 0.5)

    projection_point.x = (rho * math.cos(theta) * math.sin(phi))
    projection_point.y = (rho * math.sin(theta))
    projection_point.z = - (rho * math.cos(theta) * math.cos(phi))

    try:
        angle = vector.calculate_angle_between_vectors(left_eye_vector, camera_vectors[0])
        # print(angle)
    except ValueError:
        pass
