import flowpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

def shift_img(img, flow, alpha):
    xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    xx_shifted = (xx - (flow[:, :, 0] * alpha)).astype(np.float32)
    yy_shifted = (yy - (flow[:, :, 1] * alpha)).astype(np.float32)

    shifted_coords = np.array([yy_shifted.flatten(), xx_shifted.flatten()])
    shifted_img = np.ones_like(img)
    for d in range(img.shape[2]):
        shifted_img[:, :, d] = np.reshape(map_coordinates(img[:, :, d], shifted_coords),
                                          (img.shape[0], img.shape[1]))

    return shifted_img

flow = flowpy.flow_read("out.flo")
img = plt.imread("images/face_1_front.jpg")

min_ratio = 0.00
max_ratio = 1.00

interpolation_to_generate = np.arange(min_ratio, max_ratio + 0.01, 0.01)

for ratio in interpolation_to_generate:
    ratio = round(ratio, 2)

    interpolate = shift_img(img, flow, ratio)

    plt.imsave("out/interpolate_" + str(ratio) + "_rgb.jpg", interpolate)


fig, ax = plt.subplots()
ax.imshow(flowpy.flow_to_rgb(flow))
plt.show()