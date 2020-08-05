import vector
import matplotlib.pyplot as plt
import numpy as np
from skylibs.envmap import EnvironmentMap

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

viewing_circle_radius = 0.032
optical_centres_radius = 0.15
cameras_angle = (60., 180., 300.)

panos = []

for i in range(1, 4):
    panos.append(plt.imread('images/360render_' + str(i) + '.jpeg'))


height, width, channels = panos[0].shape

projection_point = vector.Point(1., 0., 0.)

left_eye_point = vector.EyePoint(projection_point, side='left')
right_eye_point = vector.EyePoint(projection_point, side='right')

left_eye_vector = vector.Vector(projection_point, left_eye_point)

camera_vectors = []

for camera_angle in cameras_angle:
    camera_point = vector.Point()

    camera_point.x = optical_centres_radius * np.cos(camera_angle * np.pi / 180.)
    camera_point.y = optical_centres_radius * np.sin(camera_angle * np.pi / 180.)

    camera_vectors.append(vector.Vector(projection_point, camera_point))


for i in range(width):
    # print((2048 - i) % 4096)
    phi = i * 360. / width

    projection_point.x = np.cos([np.array(phi) * np.pi / 180.])[0]
    projection_point.y = np.sin([np.array(phi) * np.pi / 180.])[0]

    angle = vector.calculate_angle_between_vectors(left_eye_vector, camera_vectors[0])


fig, ax = plt.subplots(3, 1)

env_maps = []

for i, ax in enumerate(ax.ravel()):
    env_map = EnvironmentMap('images/360render_' + str(i + 1) + '.jpeg', 'latlong')

    env_maps.append(env_map)

    ax.imshow(env_map.data)

plt.show()


fig, ax = plt.subplots(3, 1)

for i, (ax, env_map) in enumerate(zip(ax.ravel(), env_maps)):
    env_map = env_map.copy().convertTo('sphere')

    ax.imshow(env_map.data)

plt.show()

