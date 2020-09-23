import cv2
import numpy as np
import json
import random
from skimage import io
import matplotlib.pyplot as plt

OUT = "out/"


def print_type(array):
    print("min: ", np.amin(array), ", max: ", np.amax(array))
    print("shape: ", array.shape, ", type: ", array.dtype)


def cvshow(img, filename=None):
    if (np.floor(np.amax(img)) > 2):
        img = img.astype(np.uint8)

    if img.dtype is (np.dtype('float64') or np.dtype('float32')):
        img = (img * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
    if height > 1080 or width > 1920:
        cv2.resizeWindow(filename, (width // 2), (height // 2))

    #    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
    cv2.imshow(filename, img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s') and filename is not None:
        #        if(np.amax(img) <= 1):
        #            img = img*255
        #        print(np.amax(img), img.dtype)
        cv2.imwrite(OUT + filename + ".jpg", img)
    cv2.destroyAllWindows()


def cvwrite(img, filename=None, path=OUT):
    if (np.floor(np.amax(img)) > 1):
        img = img.astype(np.uint8)

    if img.dtype is (np.dtype('float64') or np.dtype('float32')):
        img = (img * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if filename is None:
        fullpath = path
    else:
        fullpath = path + filename
    cv2.imwrite(fullpath, img)


def split_cube(cube):
    w = int(cube.shape[0] / 4)
    faces = {}
    faces["top"] = cube[0:w, w:w * 2, :]
    faces["left"] = cube[w:2 * w, 0:w, :]
    faces["front"] = cube[w:2 * w, w:2 * w, :]
    faces["right"] = cube[w:2 * w, 2 * w:3 * w]
    faces["bottom"] = cube[2 * w:3 * w, w:2 * w, :]
    faces["back"] = cube[3 * w:4 * w, w:2 * w, :]
    return faces


def build_cube(faces):
    w = faces["top"].shape[0]  # width of a single face
    if len(faces["top"].shape) is 3:
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


def build_params(p=0.5, l=5, w=13, i=15, poly_expansion=7, sd=1.5, path=".", store=True):
    params = {
        "pyr_scale": p,
        "levels": l,
        "winsize": w,
        "iters": i,
        "poly_expansion": poly_expansion,
        "sd": sd
    }

    if store:
        with open(path + '/ofparams.json', 'w', encoding='utf-8') as json_file:
            json.dump(params, json_file, ensure_ascii=False, indent=4)

    return params


def load_params(path='.'):
    try:
        with open(path + '/ofparams.json', 'r') as json_file:
            params = json.load(json_file)
    except FileNotFoundError:
        params = build_params(store=False)
    return params;


def get_point_on_plane(A, B, C, dist1=None, dist2=None):
    """
    calculates a (random) point D on the plane spanned by A, B, C
    if dist1 and dist2 are None, a random value is used
    dist1 determines the distance of point AB along the vector between A and B
    dist2 determines the distance of point D along the vector between AB and C

    A    
    |       
    AB----D---C
    |
    |
    |
    B
    """
    random.seed()
    if dist1 is None:
        dist1 = random.random()
    if dist2 is None:
        dist2 = random.random()

    AB = A + dist1 * (B - A)
    return AB + dist2 * (C - AB)


def plot_points(points, points2=None, points3=None, numpoints=200):
    """
    plots 3d points with matplotlib
    points, points2, points3 are 3d arrays of points
    numpoints is the number of points to be shown (subsampling in case of too many points for useful visualization)
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    p = sample_points(points, numpoints)
    ax.scatter(p[:, :, 0], p[:, :, 1], p[:, :, 2], color='blue')
    if points2 is not None:
        p = sample_points(points2, numpoints)
        ax.scatter(p[:, :, 0], p[:, :, 1], p[:, :, 2], color='orange')
    if points3 is not None:
        p = sample_points(points3, numpoints)
        ax.scatter(p[:, :, 0], p[:, :, 1], p[:, :, 2], color='purple')

    #    ax.scatter(
    #        np.array([limit, limit, limit, limit,
    #        -limit, -limit, -limit, -limit]),
    #        np.array([limit, limit, -limit, -limit,
    #        limit, limit, -limit, -limit]),
    #        np.array([limit, -limit, limit, -limit,
    #        limit, -limit, limit, -limit])
    #    )

    plt.show()


def plot_rays(origins, targets, dim=3, numpoints=200):
    """
    plots rays from origins to targets with matplotlib
    origins and targets are 3d arrays
    dim is either 2 or 3 depending on whether the plot should be in 2d or in 3d
    numpoints is the number of points to be shown (subsampling in case of too many points for useful visualization)
    """
    o = sample_points(origins, numpoints)
    t = sample_points(targets, numpoints)
    t = t.reshape(-1, t.shape[-1])
    if origins.shape != targets.shape:
        if origins.shape[0] == 3 or origins.shape[0] == 2:
            o = np.full_like(t, o)
        else:
            raise NotImplementedError
    else:
        o = o.reshape(-1, o.shape[-1])
    diff = t - o

    plot = plt.figure()
    if dim == 3:
        ax = plt.axes(projection='3d')
        ax.set_zlabel('Z')
        plt.quiver(o[:, 0], o[:, 1], o[:, 2], diff[:, 0], diff[:, 1], diff[:, 2], arrow_length_ratio=0.1)
    else:
        ax = plt.axes()
        plt.quiver(o[:, 0], o[:, 1], diff[:, 0], diff[:, 1])
        ax.axis('equal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()


def sample_points(points, numpoints=200):
    """
    samples points from a uniform distribution
    points is a 3d array, from which the samples are taken uniformly from the width and height
    """
    if len(points.shape) == 1:
        points = points[:, np.newaxis]

    if (points.shape[0] * points.shape[1]) > numpoints:
        # find the number of points to keep for width and height so that the total points will be numpoints
        height = int(np.sqrt(numpoints / 2))
        dist_h = int(points.shape[0] / height)
        width = 2 * height
        dist_w = int(points.shape[1] / width)
        # slice the points
        return points[0:-1:dist_h, 0:-1:dist_w, :]
    else:
        return points
