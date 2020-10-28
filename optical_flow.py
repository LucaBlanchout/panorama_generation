import cv2
import numpy as np
from pathlib import Path


class OpticalFlow:
    def __init__(self, pano_1_grey, pano_2_grey, pano_index, out_path):
        self.pano_1_grey = pano_1_grey
        self.pano_2_grey = pano_2_grey
        self.pano_index = pano_index

        self.flow = None
        self.bgr = None
        self.vector_directions_image = None

        self.out_path = out_path
        Path(self.out_path + 'interpolation/').mkdir(parents=True, exist_ok=True)

        self.calculate_flow()
        self.calculate_bgr()
        self.calculate_vector_directions_image()
        self.write_flow()
        self.write_bgr()

    def calculate_flow(self):
        # self.flow = cv2.calcOpticalFlowFarneback(
        #     self.pano_1_grey,
        #     self.pano_2_grey,
        #     None,
        #     pyr_scale=0.5,
        #     levels=3,
        #     winsize=30,
        #     iterations=3,
        #     poly_n=5,
        #     poly_sigma=1.2,
        #     flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        # )

        self.flow = cv2.calcOpticalFlowFarneback(
            self.pano_1_grey,
            self.pano_2_grey,
            None,
            pyr_scale=0.5,
            levels=5,
            winsize=15,
            iterations=20,
            poly_n=7,
            poly_sigma=1.5,
            flags=0
        )

    def calculate_bgr(self):
        h, w = self.flow.shape[:2]
        fx, fy = self.flow[:, :, 0], self.flow[:, :, 1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx * fx + fy * fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v * 4, 255)
        self.bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def calculate_vector_directions_image(self, step=16):
        h, w = self.pano_2_grey.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = self.flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(self.pano_2_grey, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        self.vector_directions_image = vis

    def plot_flow(self):
        cv2.imshow('img', self.vector_directions_image)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()

    def plot_bgr(self):
        cv2.imshow('img', self.bgr)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()

    def write_flow(self):
        cv2.imwrite(self.out_path + 'flow_' + str(self.pano_index[0]) + '_' + str(self.pano_index[1]) + '.jpg',
                    self.vector_directions_image)

    def write_bgr(self):
        cv2.imwrite(self.out_path + 'bgr_' + str(self.pano_index[0]) + '_' + str(self.pano_index[1]) + '.jpg',
                    self.bgr)
