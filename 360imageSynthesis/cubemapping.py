import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

import utils
from envmap import EnvironmentMap, rotation_matrix

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
FACES = ['top', 'front', 'left', 'right', 'bottom', 'back']


class ExtendedCubeMap:
    """
    Creates an extended cube: each face is extended so that points moving across edges can be tracked correctly by optical flow algorithms
    Input can either be a panorama in the format latlong or cube or a synthesized extended cube
    If it is a regular panorama, it is extended and the extended faces are stored
    If it is an extended cube, it is not extended but stored as is
    """

    def __init__(self, imgpath, format, percent=1.2, fov=90, w_original=None):
        self.fov = fov
        self.format = format

        f = "cube" if format is "Xcube" else format

        self._envMap = EnvironmentMap(imgpath, f)
        self.w = int(self._envMap.data.shape[0] / 4)  # width of cube face

        # Xcube -> already an extended cube
        if format is "Xcube":
            self.extended = utils.split_cube(imgpath)
            self.w_original = w_original
        else:
            self.extended = self.extend_projection(105)
            self.w_original = self.w
            self.w = self.extended["front"].shape[0]
            if format is not 'cube':
                self._envMap.convertTo('cube')

#        for face in FACES:
#            utils.cvshow(self.extended[face])
#        utils.cvshow(self.get_Xcube())

    def get_Xcube(self):
        #        return self._envMap.data
        return utils.build_cube(self.extended)

    def calc_clipped_cube(self):
        """
        calculates the original, non-extended cube
        TODO: does not return the exact original cube!
        """
        if self.format is "Xcube":
            faces = {}
            border_width = int(np.floor((self.w - self.w_original) / 2))
            # print(border_width)

            xx, yy = np.meshgrid(np.arange(self.w_original), np.arange(self.w_original))
            clipped_coords = np.array([yy.flatten(), xx.flatten()])
            depth = self._envMap.data.shape[2]

            for face in FACES:
                clipped = self.extended[face][border_width:-border_width, border_width:-border_width, :]
                faces[face] = np.zeros((self.w_original, self.w_original, depth))
                for d in range(depth):
                    faces[face][:, :, d] = np.reshape(map_coordinates(clipped[:, :, d], clipped_coords),
                                                      (self.w_original, self.w_original))
            # print(faces["front"].shape)
            return utils.build_cube(faces)

#            fov = 2*np.rad2deg(np.arctan(self.w_original/self.w))
#            print("new fov", fov)
#            faces = self.extend_projection(fov)
#            return utils.build_cube(faces)
        else:
            return self._envMap.data

    def extend_projection(self, fov):
        """
        calculates a projection for each face of the cube with the given field of view 
        """
        # adjacent is 1 (unit sphere) --> opposite is tan(fov)
        norm_original_width = np.tan(np.deg2rad(self.fov / 2)) * 2
        # print("norm original", norm_original_width)
        norm_new_width = np.tan(np.deg2rad(fov / 2)) * 2
        # print("norm new", norm_new_width)

        face_width = int(self.w * (norm_new_width / norm_original_width))

        # print("old face width", self.w)
        # print("new face width", face_width)
        # face_width = int(self.w_original * 1.1) #TODO make dependent on fov

        rotations = {"top": rotation_matrix(0, np.deg2rad(-90), 0),
                     "front": rotation_matrix(0, 0, 0),
                     "left": rotation_matrix(np.deg2rad(-90), 0, 0),
                     "right": rotation_matrix(np.deg2rad(90), 0, 0),
                     "bottom": rotation_matrix(0, np.deg2rad(90), 0),
                     "back": rotation_matrix(0, np.deg2rad(180), 0)
                     }
        faces = {}
        for face in FACES:
            faces[face] = self._envMap.project(fov, rotations[face], 1., (face_width, face_width))
        return faces

    def optical_flow(self, other, flowfunc):
        """
        applies an optical flow algorithm on each face of the extended cube
        other: other ExtendedCubeMap for flow calculation
        flowfunc: optical flow function returning a 2D array of vectors
        """
        flow = {}
        for face in FACES:
            flow[face] = flowfunc(self.extended[face].astype(np.uint8), other.extended[face].astype(np.uint8))

        flow_cube = utils.build_cube(flow)
        return flow_cube

    def optical_flow_face(self, face, other, flowfunc):
        """
        for testing purposes
        applies an optical flow algorithm only on the front face of the extended cube
        other: other ExtendedCubeMap for flow calculation
        flowfunc: optical flow function returning a 2D array of vectors
        """
        return flowfunc(self.extended[face], other.extended[face])
