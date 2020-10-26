import cv2
import numpy as np
import itertools


# def plot_opencv_image(img):
#     cv2.imshow('img', img)
#     k = cv2.waitKey(0)
#     if k == 27:  # wait for ESC key to exit
#         cv2.destroyAllWindows()
#
#
# def draw_flow(img, flow, step=16):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
#     fx, fy = flow[y, x].T
#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines + 0.5)
#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     cv2.polylines(vis, lines, 0, (0, 255, 0))
#     for (x1, y1), (_x2, _y2) in lines:
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
#     return vis
#
#
# def draw_hsv(flow):
#     h, w = flow.shape[:2]
#     fx, fy = flow[:, :, 0], flow[:, :, 1]
#     ang = np.arctan2(fy, fx) + np.pi
#     v = np.sqrt(fx*fx + fy*fy)
#     hsv = np.zeros((h, w, 3), np.uint8)
#     hsv[..., 0] = ang*(180/np.pi/2)
#     hsv[..., 1] = 255
#     hsv[..., 2] = np.minimum(v*4, 255)
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     return bgr
#
#
# def warp_flow(img, flow):
#     h, w = flow.shape[:2]
#     flow = -flow
#     flow[:, :, 0] += np.arange(w)
#     flow[:, :, 1] += np.arange(h)[:, np.newaxis]
#     res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
#     return res

# def calculate_optical_flows_between_panoramas(panos, out_path):
#     pano_greyscale = []
#     for pano in panos:
#         pano_grey = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
#         pano_greyscale.append(pano_grey)
#
#     pano_permutation_indices = list(itertools.permutations(range(len(pano_greyscale)), 2))
#
#     for permutation_index in pano_permutation_indices:
#         pano_1_grey = pano_greyscale[permutation_index[0]]
#         pano_2_grey = pano_greyscale[permutation_index[1]]
#
#         cur_glitch = panos[permutation_index[0]].copy()
#
#         flow = cv2.calcOpticalFlowFarneback(
#             pano_1_grey,
#             pano_2_grey,
#             None,
#             pyr_scale=0.5,
#             levels=3,
#             winsize=30,
#             iterations=3,
#             poly_n=5,
#             poly_sigma=1.2,
#             flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
#         )
#
#         cv2.imwrite(
#             out_path + 'optical_flow/flow_' + str(permutation_index[0]) + '_' + str(permutation_index[1]) + '.jpg',
#             draw_flow(pano_2_grey, flow))
#         cv2.imwrite(
#             out_path + 'optical_flow/hsv_' + str(permutation_index[0]) + '_' + str(permutation_index[1]) + '.jpg',
#             draw_hsv(flow))
#         cv2.imwrite(
#             out_path + 'optical_flow/glitch_' + str(permutation_index[0]) + '_' + str(permutation_index[1]) + '.jpg',
#             warp_flow(cur_glitch, flow))


# def calculate_optical_flows_between_panoramas(base_panoramas):
#     panos_permutation_indexes = list(itertools.permutations(range(len(base_panoramas)), 2))
#
#     for idx in panos_permutation_indexes:
#         base_pano_1 = base_panoramas[idx[0]]
#         base_pano_2 = base_panoramas[idx[1]]
#
#         pano_1_grey = base_pano_1.grey_img
#         pano_2_grey = base_pano_2.grey_img
#
#         cur_glitch = base_pano_1.bgr_img.copy()
#
#         flow = cv2.calcOpticalFlowFarneback(
#             pano_1_grey,
#             pano_2_grey,
#             None,
#             pyr_scale=0.5,
#             levels=3,
#             winsize=30,
#             iterations=3,
#             poly_n=5,
#             poly_sigma=1.2,
#             flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
#         )
#
#         base_pano_1.optical_flows[idx[1]] = OpticalFlow(flow)


class OpticalFlow:
    def __init__(self, pano_1_grey, pano_2_grey, pano_index):
        self.pano_1_grey = pano_1_grey
        self.pano_2_grey = pano_2_grey
        self.pano_index = pano_index

        self.flow = None
        self.bgr = None
        self.vector_directions_image = None
        self.calculate_flow()
        self.calculate_bgr()
        self.calculate_vector_directions_image()

    def calculate_flow(self):
        self.flow = cv2.calcOpticalFlowFarneback(
            self.pano_1_grey,
            self.pano_2_grey,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=30,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
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

    def write_flow(self, path):
        cv2.imwrite(path + 'flow_' + str(self.pano_index[0]) + '_' + str(self.pano_index[1]) + '.jpg',
                    self.vector_directions_image)

    def write_bgr(self, path):
        cv2.imwrite(path + 'bgr_' + str(self.pano_index[0]) + '_' + str(self.pano_index[1]) + '.jpg',
                    self.bgr)
