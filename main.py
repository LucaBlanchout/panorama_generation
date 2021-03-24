from base_panorama import BasePanorama, BasePanoramaContainer
from generated_panorama import GeneratedPanoramaContainer
from camera import Camera, CameraContainer

import numpy as np
import math


np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
np.seterr(divide='ignore', invalid='ignore')

width_resolution = 1024
number_of_cameras = 3
cameras_to_keep = 2
# envmap_type = 'latlong'
envmap_type = 'cube'

in_path = 'images/' + str(width_resolution) + "/" + str(number_of_cameras) + '/real/360render_'
base_out_path = "out/" + envmap_type + '/' + str(width_resolution) + "/" + str(number_of_cameras) + "/keep_" + str(cameras_to_keep) + "/real/"

base_panorama_container = BasePanoramaContainer(base_out_path=base_out_path)
camera_container = CameraContainer()

for i in range(number_of_cameras):
    camera_angle = (1 + 2 * i) * math.pi / number_of_cameras + math.pi / 2

    camera_container.append(
        Camera(
            camera_angle
        )
    )

    base_panorama_container.append(
        BasePanorama(
            i,
            in_path,
            base_out_path,
            envmap_type=envmap_type
        )
    )

base_panorama_container.write_base_panoramas()

generated_panorama_container = GeneratedPanoramaContainer(
    base_panorama_container,
    camera_container,
    cameras_to_keep,
    envmap_type,
    base_out_path
)

rho_range = np.linspace(0.5, 5, 10)
for rho in rho_range:
    print("\nStarting rho =", rho)

    generated_panorama_container.set_rho_and_generate_panoramas(rho)

    print("Finished rho=", rho)
