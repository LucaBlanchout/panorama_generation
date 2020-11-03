import utils

import cv2
import numpy as np
from pathlib import Path


class OpticalFlow:
    def __init__(self, pano_1_extended_cubemap, pano_2_extended_cubemap, pano_index, out_path):
        self.pano_index = pano_index

        self.pano_1_extended_cubemap = pano_1_extended_cubemap
        self.pano_2_extended_cubemap = pano_2_extended_cubemap
        # Needs to be inverted for calculate_vector_directions_image()
        self.panos_grey = [
            cv2.cvtColor(np.float32(self.pano_2_extended_cubemap.get_extended_cube()), cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(np.float32(self.pano_1_extended_cubemap.get_extended_cube()), cv2.COLOR_RGB2GRAY),
        ]

        # These three properties also have their inverted counterpart.
        self.flows = [None, None]
        self.vector_directions_images = [None, None]
        self.bgrs = [None, None]

        self.out_path = out_path
        Path(self.out_path + 'interpolation/').mkdir(parents=True, exist_ok=True)

        self.calculate_flow()
        self.calculate_bgr()
        self.calculate_vector_directions_image()

        self.write_flow()
        self.write_bgr()

    def calculate_flow(self):
        flow = {}
        inv_flow = {}
        for face in utils.FACES:
            grey_face_1 = cv2.cvtColor(self.pano_1_extended_cubemap.extended[face].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            grey_face_2 = cv2.cvtColor(self.pano_2_extended_cubemap.extended[face].astype(np.uint8), cv2.COLOR_RGB2GRAY)

            flow[face] = cv2.calcOpticalFlowFarneback(
                grey_face_1,
                grey_face_2,
                None,
                pyr_scale=0.5,
                levels=5,
                winsize=15,
                iterations=20,
                poly_n=7,
                poly_sigma=1.5,
                flags=0
            )

            inv_flow[face] = cv2.calcOpticalFlowFarneback(
                grey_face_2,
                grey_face_1,
                None,
                pyr_scale=0.5,
                levels=5,
                winsize=15,
                iterations=20,
                poly_n=7,
                poly_sigma=1.5,
                flags=0
            )

        flow = utils.build_cube(flow)
        inv_flow = utils.build_cube(inv_flow)

        self.flows = [flow, inv_flow]

    def calculate_bgr(self):
        for i in range(len(self.flows)):
            h, w = self.flows[i].shape[:2]
            fx, fy = self.flows[i][:, :, 0], self.flows[i][:, :, 1]
            ang = np.arctan2(fy, fx) + np.pi
            v = np.sqrt(fx * fx + fy * fy)
            hsv = np.zeros((h, w, 3), np.uint8)
            hsv[..., 0] = ang * (180 / np.pi / 2)
            hsv[..., 1] = 255
            hsv[..., 2] = np.minimum(v * 4, 255)
            self.bgrs[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def calculate_vector_directions_image(self, step=16):
        for i in range(len(self.panos_grey)):
            h, w = self.panos_grey[i].shape[:2]
            y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
            fx, fy = self.flows[i][y, x].T
            lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)
            vis = cv2.cvtColor(self.panos_grey[i], cv2.COLOR_GRAY2BGR)
            cv2.polylines(vis, lines, 0, (0, 255, 0))
            for (x1, y1), (_x2, _y2) in lines:
                cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
            self.vector_directions_images[i] = vis

    def plot_flow(self):
        cv2.imshow('vdi', self.vector_directions_images[0])
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()

        cv2.imshow('ivdi', self.vector_directions_images[1])
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()

    def plot_bgr(self):
        cv2.imshow('vgr', self.bgrs[0])
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()

        cv2.imshow('ibgr', self.bgrs[1])
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()

    def write_flow(self):
        cv2.imwrite(self.out_path + 'flow_' + str(self.pano_index[0]) + '_' + str(self.pano_index[1]) + '.jpg',
                    self.vector_directions_images[0])

        cv2.imwrite(self.out_path + 'flow_' + str(self.pano_index[1]) + '_' + str(self.pano_index[0]) + '.jpg',
                    self.vector_directions_images[1])

    def write_bgr(self):
        cv2.imwrite(self.out_path + 'bgr_' + str(self.pano_index[0]) + '_' + str(self.pano_index[1]) + '.jpg',
                    self.bgrs[0])

        cv2.imwrite(self.out_path + 'bgr_' + str(self.pano_index[1]) + '_' + str(self.pano_index[0]) + '.jpg',
                    self.bgrs[1])
