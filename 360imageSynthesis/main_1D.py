import cv2
import numpy as np
import matplotlib.pyplot as plt

import utils
from interpolation import Interpolator1D

"""
### planar interpolation ###
imgAfront = cv2.cvtColor(cv2.imread("../../data/1D_testsets/01_exterior_stadium/front_A.jpg", 1), cv2.COLOR_BGR2RGB)
imgBfront = cv2.cvtColor(cv2.imread("../../data/1D_testsets/01_exterior_stadium/front_B.jpg", 1), cv2.COLOR_BGR2RGB)
interpolator_p = Interpolator1D(imgAfront, imgBfront, "planar")
out_p = interpolator_p.interpolate(0.5)
flow_p = interpolator_p.get_flow_visualization()
utils.cvshow(out_p, 'planar_front.jpg')
utils.cvshow(flow_p, 'planar_front_flow.jpg')
"""

### panoramic interpolation ###
# imgA = cv2.cvtColor(cv2.imread("1D_testsets/01_exterior_stadium/03.JPG", 1), cv2.COLOR_BGR2RGB)
# imgB = cv2.cvtColor(cv2.imread("1D_testsets/01_exterior_stadium/02.JPG", 1), cv2.COLOR_BGR2RGB)
imgA = cv2.cvtColor(cv2.imread("images/360render_2.jpg", 1), cv2.COLOR_BGR2RGB)
imgB = cv2.cvtColor(cv2.imread("images/360render_0.jpg", 1), cv2.COLOR_BGR2RGB)

# imgA = cv2.cvtColor(cv2.imread("../../data/1D_testsets/02_meeting_room/20.jpg", 1), cv2.COLOR_BGR2RGB)
# imgB = cv2.cvtColor(cv2.imread("../../data/1D_testsets/02_meeting_room/21.jpg", 1), cv2.COLOR_BGR2RGB)

interpolator = Interpolator1D(imgA, imgB, "latlong")

# visualize the flow
flowcube = interpolator.get_flow_visualization()
utils.cvwrite(flowcube, 'flow_cube.jpg')

cv2.imshow('Optical flow', flowcube)
cv2.waitKey(0)
cv2.destroyAllWindows()


# visualize the difference between the original cube and the extended cube
utils.cvwrite(interpolator.A.calc_clipped_cube(), '360render_2_original.jpg')
utils.cvwrite(interpolator.A.get_Xcube(), '360render_2_extended.jpg')
utils.cvwrite(interpolator.B.calc_clipped_cube(), '360render_0_original.jpg')
utils.cvwrite(interpolator.B.get_Xcube(), '360render_0_extended.jpg')

# interpolate a specific position
# out = interpolator.interpolate(0.5)
# utils.cvwrite(out, '02_imgInterpolated_clipped_0.5.jpg')

# interpolate on the line between the two viewpoints
# for d in np.around(np.linspace(0, 1, 11), 1):
for d in [0.5]:
    out = interpolator.interpolate(d)
    utils.cvwrite(out, '02_imgInterpolated_' + str(d) + '.jpg')

# visualize the difference between the original (clipped_in), the extended, and the clipped cube (At the moment clipped_in (original) and clipped_out (output of calc_clipped_cube) are not identical. This needs to be fixed)
# utils.cvshow(interpolator.A.calc_clipped_cube(), 'clipped_in')
# utils.cvshow(utils.build_cube(interpolator.A.extended), 'clipped_extended')
# utils.cvshow(interpolator.out.calc_clipped_cube(), 'clipped_out')
