import numpy as np
from scipy.spatial.transform import Rotation

import utils

dirname = "normalized"
rawname = "raw"

def parse_metadata(txt):
    """
    takes a text file as an input containing axis information, and rotation and position metadata and returns the rotations and positions as np arrays
    input must have rotation as quaternion and position for each image on a separate line, in order, separated by commas
    the first two lines must be axis order and axis sign respectively (in case the coordinate system of the capture is different from the one used by CaptureSet, xy is "floor" plane)
    axis_order: positions to shift the axes, eg from x,y,z to x,z,y would be [0,1,-1]
    axis_sign: if any of the axes are flipped, flip them back: eg from x,y,z to x,-y,z would be [1,-1,1]
    """
    rotations = []
    positions = []
    with open(txt, 'r') as file:
        lines = file.readlines()
        axis_order = np.array(lines[0].split(',')).astype(int)
        axis_sign = np.array(lines[1].split(',')).astype(int)
        print("order", axis_order)
        print("sign", axis_sign)
        for i in range(2,len(lines)):
            print(lines[i])
            line = lines[i]
            strs = line.split(',')
            if len(strs) is not 7:
                raise Exception("One or more of the lines in the input file have the wrong number of values")

            rot = np.array([strs[0], strs[1], strs[2], strs[3]]).astype(float)
            rotations.append(rot)

            pos = np.array([
                axis_sign[0 + axis_order[0]] * float(strs[ 4 + axis_order[0] ]),
                axis_sign[1 + axis_order[0]] * float(strs[ 5 + axis_order[1] ]),
                axis_sign[2 + axis_order[0]] * float(strs[ 6 + axis_order[2] ])
                ])
#            pos = np.array([strs[4], strs[6], strs[5]]).astype(float) #so that x/y plane is parallel to floor
            positions.append(pos)

    return np.array(positions), np.array(rotations)

#TODO make center and normalize the same structure
def center(points):
    """
    takes a point cloud of numpy arrays
    returns same point cloud centered around 0,0,0
    """
#   normalize
#    minima = np.amin(points)
#    points -= minima
#
#    maxima = np.amax(points)
#    points = ( points/maxima ) * suggested_diameter

    minima = np.amin(points, axis=0)
    maxima = np.amax(points, axis=0)
    center_point = minima + (maxima-minima) * 0.5
    shifted_points = points - center_point

    return shifted_points

def normalize_rotation(capture_set):
    """
    takes a capture set and rotates the captures so that they all have the same rotation (that of the first capture)
    """
    for i in range(1, capture_set.get_size()):
        capture = capture_set.get_capture(i)
        rotation = Rotation.from_quat([capture.rot[0], capture.rot[1], capture.rot[2], capture.rot[3]]) #TODO find out why rotation is not interpreted as an array
        capture.rotate(rotation.inv())
        print("Rotating " + capture.imgpath)
        capture.store_image()
        new_rot = rotation * rotation.inv()
        capture_set.rotations[i] = new_rot.as_quat()
    capture_set.store_rotations()


