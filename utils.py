import numpy as np
from scipy.ndimage.interpolation import map_coordinates


FACES = ['top', 'front', 'left', 'right', 'bottom', 'back']


def split_cube(cube):
    w = int(cube.shape[0] / 4)

    faces = {
        "top": cube[0:w, w:w * 2, :],
        "left": cube[w:2 * w, 0:w, :],
        "front": cube[w:2 * w, w:2 * w, :],
        "right": cube[w:2 * w, 2 * w:3 * w],
        "bottom": cube[2 * w:3 * w, w:2 * w, :],
        "back": cube[3 * w:4 * w, w:2 * w, :]
    }

    return faces


def build_cube(faces):
    shape = faces["top"].shape

    w = shape[0]

    if len(shape) == 3:
        cube = np.zeros((w * 4, w * 3, faces["top"].shape[2]))
        cube[0:w, w:w * 2, :] = faces["top"]
        cube[w:2 * w, 0:w, :] = faces["left"]
        cube[w:2 * w, w:2 * w, :] = faces["front"]
        cube[w:2 * w, 2 * w:3 * w, :] = faces["right"]
        cube[2 * w:3 * w, w:2 * w, :] = faces["bottom"]
        cube[3 * w:4 * w, w:2 * w, :] = faces["back"]
    else:
        cube = np.zeros_like((w * 4, w * 3))
        cube[0:w, w:w * 2] = faces["top"]
        cube[w:2 * w, 0:w] = faces["left"]
        cube[w:2 * w, w:2 * w] = faces["front"]
        cube[w:2 * w, 2 * w:3 * w] = faces["right"]
        cube[2 * w:3 * w, w:2 * w] = faces["bottom"]
        cube[3 * w:4 * w, w:2 * w] = faces["back"]

    return cube


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
