import cv2
import numpy as np
import itertools


def plot_opencv_image(img):
    cv2.imshow('img', img)
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx + fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang*(180/np.pi/2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def draw_hsv_2(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3))

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


# TODO : Don't calculate twice every optical flow
def calculate_optical_flows_between_panoramas(panos, out_path):
    pano_greyscale = []
    for pano in panos:
        pano_grey = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
        pano_greyscale.append(pano_grey)

    pano_permutation_indices = list(itertools.permutations(range(len(pano_greyscale)), 2))

    for permutation_index in pano_permutation_indices:
        pano_1_grey = pano_greyscale[permutation_index[0]]
        pano_2_grey = pano_greyscale[permutation_index[1]]

        cur_glitch = panos[permutation_index[0]].copy()

        flow = cv2.calcOpticalFlowFarneback(
            pano_1_grey,
            pano_2_grey,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=400,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )

        cv2.imwrite(out_path + 'optical_flow/flow_' + str(permutation_index[0]) + '_' + str(permutation_index[1]) + '.jpg', draw_flow(pano_2_grey, flow))
        cv2.imwrite(out_path + 'optical_flow/hsv_' + str(permutation_index[0]) + '_' + str(permutation_index[1]) + '.jpg', draw_hsv(flow))
        cv2.imwrite(out_path + 'optical_flow/glitch_' + str(permutation_index[0]) + '_' + str(permutation_index[1]) + '.jpg', warp_flow(cur_glitch, flow))
