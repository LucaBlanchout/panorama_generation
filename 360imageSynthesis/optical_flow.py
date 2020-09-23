import cv2
import numpy as np

import utils


# based on https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

def farneback_of(imgA, imgB, param_path="."):
    """
    uses the OpenCV implementation of the Farneback algorithm for calculating optical flow
    input: two uint8 images that the optical flow should be calculated on, flow is calculated from A to B
    param_path: if specific optical flow parameters should be used, pass the file (which was created by utils.build_params)

    returns: a numpy array of flow vectors of the same width and height as the input images
    """
    if imgA is None or imgB is None:
        raise Exception("Image(s) not loaded correctly. Aborting")

    if imgA.dtype != np.uint8:
        raise Exception("Image A needs to be type uint8. It is currently " + str(imgA.dtype))

    if imgB.dtype != np.uint8:
        raise Exception("Image B needs to be type uint8. It is currently " + str(imgB.dtype))

    # check whether greyscale, if not, convert
    if len(imgA.shape) > 2:
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    if len(imgB.shape) > 2:
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # if file can't be loaded, uses default parameters
    params = utils.load_params(param_path)

    flow = cv2.calcOpticalFlowFarneback(imgA, imgB, None, params['pyr_scale'], params['levels'], params['winsize'],
                                        params['iters'], params['poly_expansion'], params['sd'], 0)

    return flow


def visualize_flow(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr


def farneback_of_test(imgA, imgB):
    """
    test flow parameters (this is quick and dirty)
    """
    if imgA is None or imgB is None:
        raise Exception("Image(s) not loaded correctly. Aborting")

    if imgA.dtype != np.uint8:
        raise Exception("Image A needs to be type uint8. It is currently " + str(imgA.dtype))

    if imgB.dtype != np.uint8:
        raise Exception("Image B needs to be type uint8. It is currently " + str(imgB.dtype))

    # check whether greyscale, if not, convert
    if len(imgA.shape) > 2:
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    if len(imgB.shape) > 2:
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros((imgA.shape[0], imgA.shape[1], 3))
    hsv[..., 1] = 255

    pyr_scale = [0.5]
    levels = [3, 5]
    winsize = [13, 15, 20]
    iters = [5, 10, 15]
    poly = [(7, 1.5)]

    for pyr in pyr_scale:
        for l in levels:
            for w in winsize:
                for i in iters:
                    for p in poly:
                        flow = cv2.calcOpticalFlowFarneback(imgA, imgB, None, pyr, l, w, i, p[0], p[1], 0)

                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        hsv[..., 0] = ang * 180 / np.pi / 2
                        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                        cv2.imwrite(
                            "../../data/out/" + str(pyr) + "l" + str(l) + "w" + str(w) + "i" + str(i) + "p" + str(
                                p) + ".jpg", bgr)

    return flow, bgr
