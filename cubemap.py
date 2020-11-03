import utils
from skylibs.envmap import EnvironmentMap, rotation_matrix
from skylibs.envmap.projections import cube2world

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt

"""
sides of the cube:
         ___
        | T |
     ___|___|___
    | L | F | R |
    |___|___|___|
        |BO |
        |___|
        |BA |
        |___|
T: top
L: left
F: front
R: right
BO: bottom
BA: back
"""


class ExtendedCubeMap:
    def __init__(self, img, envmap_type, fov=90, w_original=None):
        self.fov = fov
        self.envmap_type = envmap_type

        f = "cube" if self.envmap_type == "Xcube" else self.envmap_type

        self._envmap = EnvironmentMap(img, f)
        self.w = int(self._envmap.data.shape[0] / 4)

        if envmap_type == "Xcube":
            self.extended = utils.split_cube(img)
            self.w_original = w_original
        else:
            self.extended = self.extend_projection(110)
            self.w_original = self.w
            self.w = self.extended["front"].shape[0]
            if self.envmap_type != 'cube':
                self._envmap.convertTo('cube')

        self.world_coordinates(110)

    def get_extended_cube(self):
        return utils.build_cube(self.extended)

    def get_clipped_cube(self):
        if self.envmap_type == "Xcube":
            faces = {}
            border_width = int(np.floor((self.w - self.w_original) / 2))

            xx, yy = np.meshgrid(np.arange(self.w_original), np.arange(self.w_original))
            clipped_coords = np.array([yy.flatten(), xx.flatten()])
            depth = self._envmap.data.shape[2]

            for face in utils.FACES:
                clipped = self.extended[face][border_width:-border_width, border_width:-border_width, :]
                faces[face] = np.zeros((self.w_original, self.w_original, depth))
                for d in range(depth):
                    faces[face][:, :, d] = np.reshape(map_coordinates(clipped[:, :, d], clipped_coords),
                                                      (self.w_original, self.w_original))

            return utils.build_cube(faces)
        else:
            return self._envmap.data

    def extend_projection(self, fov):
        norm_original_width = np.tan(np.deg2rad(self.fov / 2)) * 2

        norm_new_width = np.tan(np.deg2rad(fov / 2)) * 2

        face_width = int(self.w * (norm_new_width / norm_original_width))

        print("Face width = ", face_width)

        rotations = {
            "top": rotation_matrix(0, np.deg2rad(-90), 0),
            "front": rotation_matrix(0, 0, 0),
            "left": rotation_matrix(np.deg2rad(-90), 0, 0),
            "right": rotation_matrix(np.deg2rad(90), 0, 0),
            "bottom": rotation_matrix(0, np.deg2rad(90), 0),
            "back": rotation_matrix(0, np.deg2rad(180), 0)
        }

        faces = {}
        for face in utils.FACES:
            faces[face] = self._envmap.project(fov, rotations[face], 1., (face_width, face_width))
        return faces

    def world_coordinates(self, fov):
        u, v = self._envmap.imageCoordinates()
        x_base, y_base, z_base, valid_base = cube2world(u, v)

        base_world_coord = np.stack((x_base, y_base, z_base, valid_base), axis=2)

        base_world_coord_envmap = EnvironmentMap(base_world_coord, 'cube')

        face_width = self.extended['top'].shape[0]
        # norm_original_width = np.tan(np.deg2rad(self.fov / 2)) * 2
        #
        # norm_new_width = np.tan(np.deg2rad(fov / 2)) * 2
        #
        # face_width = int(self.w * (norm_new_width / norm_original_width))

        print("face_width world_coord =", face_width)

        rotations = {
            "top": rotation_matrix(0, np.deg2rad(-90), 0),
            "front": rotation_matrix(0, 0, 0),
            "left": rotation_matrix(np.deg2rad(-90), 0, 0),
            "right": rotation_matrix(np.deg2rad(90), 0, 0),
            "bottom": rotation_matrix(0, np.deg2rad(90), 0),
            "back": rotation_matrix(0, np.deg2rad(180), 0)
        }

        plt.imshow(self.get_extended_cube().astype(np.uint8))
        plt.show()

        plt.imshow(base_world_coord_envmap.data[..., 0])
        plt.show()

        faces = {}
        for face in utils.FACES:
            faces[face] = base_world_coord_envmap.project(fov, rotations[face], 1., (face_width, face_width))

        world_coord = utils.build_cube(faces)

        plt.imshow(world_coord[..., 0])
        plt.show()

        # base_face_width = x_base.shape[0] // 4
        # extended_face_width = self.extended['top'].shape[0]
        #
        # width_diff = extended_face_width - base_face_width
        # width_diff_offset = width_diff // 2
        #
        # rows = self.extended['top'].shape[0] * 4
        # cols = self.extended['top'].shape[0] * 3
        #
        # extended_world_coord = np.empty((rows, cols, 4))
        #
        # extended_world_coord[width_diff_offset:extended_face_width - width_diff_offset, extended_face_width + width_diff_offset:extended_face_width * 2 - width_diff_offset, :] = base_world_coord[0:base_face_width, base_face_width:base_face_width * 2, :]
        # extended_world_coord[extended_face_width + width_diff_offset:extended_face_width * 2 - width_diff_offset, width_diff_offset:extended_face_width - width_diff_offset, :] = base_world_coord[base_face_width:base_face_width * 2, 0:base_face_width, :]
        # extended_world_coord[extended_face_width + width_diff_offset:extended_face_width * 2 - width_diff_offset, extended_face_width + width_diff_offset:extended_face_width * 2 - width_diff_offset, :] = base_world_coord[base_face_width:base_face_width * 2, base_face_width:base_face_width * 2, :]
        # extended_world_coord[extended_face_width + width_diff_offset:extended_face_width * 2 - width_diff_offset, extended_face_width * 2 + width_diff_offset:extended_face_width * 3 - width_diff_offset, :] = base_world_coord[base_face_width:base_face_width * 2, base_face_width * 2:base_face_width * 3, :]
        # extended_world_coord[extended_face_width * 2 + width_diff_offset:extended_face_width * 3 - width_diff_offset, extended_face_width + width_diff_offset:extended_face_width * 2 - width_diff_offset, :] = base_world_coord[base_face_width * 2:base_face_width * 3, base_face_width:base_face_width * 2, :]
        # extended_world_coord[extended_face_width * 3 + width_diff_offset:extended_face_width * 4 - width_diff_offset, extended_face_width + width_diff_offset:extended_face_width * 2 - width_diff_offset, :] = base_world_coord[base_face_width * 3:base_face_width * 4, base_face_width:base_face_width * 2, :]
        #
        #
        # plt.imshow(extended_world_coord[..., 1])
        # plt.show()
        #
        # plt.imshow(base_world_coord[..., 1])
        # plt.show()
        #
        # raise SystemExit(0)
        # return x, y, z, valid
