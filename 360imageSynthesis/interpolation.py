import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt

from envmap import EnvironmentMap, projections

import utils
from optical_flow import farneback_of, visualize_flow
from cubemapping import ExtendedCubeMap

types = ["latlong", "cube", "planar"]


# TODO create Interpolator parent class if appropriate (actually, Interpolator1D should probably be a parent class with its children being Interpolator1DPlanar and Interpolator1DPanoramic
class Interpolator1D:
    """
    creates an interpolator for either planar or panoramic images
    """

    def __init__(self, imgA, imgB, type="latlong", flowfunc=farneback_of):
        """
        Initializes the interpolator with ExtendedCubeMaps and calculates the flow between the two images.

        imgA, imgB: the viewpoints (as RGB images) used in the interpolation
        type: one of [latlong | cube | planar]
        flowfunc: the flow function to be used (at the moment, only farneback_of is implemented)
        """
        self.type = type
        if type == "planar":
            self.A = imgA
            self.B = imgB
            self.flow = flowfunc(self.A, self.B)
        else:
            self.A = ExtendedCubeMap(imgA, type)
            self.B = ExtendedCubeMap(imgB, type)
            self.flow = self.A.optical_flow(self.B, flowfunc)
        self.out = None

    def clear(self):
        self.out = None

    def interpolate(self, alpha):
        """
        Uses the flow vectors to shift and blend the images to synthesize point alpha between viewpoint A and B

        alpha: distance along the vector between A and B

        returns: the blended image

        Note: self.out stores the blended image as an ExtendedCubeMap until the next time interpolate is called
        """
        self.clear()

        shifted_A = shift_img(self.get_image(self.A), self.flow, alpha)
        shifted_B = shift_img(self.get_image(self.B), -self.flow, (1 - alpha))

        out = (1 - alpha) * shifted_A + alpha * shifted_B

        if self.type is "planar":
            self.out = out
        else:
            self.out = ExtendedCubeMap(out, "Xcube", fov=self.A.fov, w_original=self.A.w_original)
            out = self.out.calc_clipped_cube()
        return out

    def get_image(self, A):
        """
        hack to be able to differentiate between ExtendedCubeMaps and planar images
        this should become obsolete once there is a distinct class for the two cases
        """
        if self.type is "planar":
            return A
        else:
            return A.get_Xcube()

    def get_flow_visualization(self):
        return visualize_flow(self.flow)


class Interpolator3D:
    def __init__(self, capture_set):
        self.dev_angles = None
        self.distances = None
        self.uvs = {}
        self.capture_set = capture_set
        self.indices = []
        self.best_indices = None
        self.point = None
        self.intersections = None

    def clear(self):
        self.dev_angles = None
        self.distances = None
        self.uvs = {}
        self.best_indices = None

    def interpolate(self, indices, point, knn=1):
        """
        Synthesizes a point from the specified viewpoints

        indices: viewpoints of the capture set to use
        point: point to be synthesized
        knn: the k nearest neighbors (based on deviation angle) to be used for blending the image
        returns: the synthesized image of the desired viewpoint

        Note: This function stores information in the class such as scene intersection points and deviation angles so that they can be visualized later on. This information is reset at each new interpolation

        """
        # reset and fill the interpolation information for later use
        self.clear()
        print("synthesizing ", point)
        print("using ", indices, " for synthesis")
        self.indices = indices
        self.point = point
        dimensions = self.capture_set.get_capture(indices[0]).img.shape
        self.dev_angles = np.full((dimensions[0], dimensions[1], len(indices)), np.pi)
        # get the rays from the new point that correspond to the uv coordinates of an equirectangular image at that point (points on the unit sphere in the point's local coordinate system)
        u, v = calc_uv_coordinates(dimensions[0])
        nx, nz, ny, _ = projections.latlong2world(u,
                                                  v)  # switch y and z because projections uses a different representation
        targets = np.dstack((nx, ny, nz))

        # calculate the intersections from the new point with the scene
        self.intersections = self.capture_set.calc_ray_intersection(point, targets)
        #        self.capture_set.draw_scene(indices=[], s_points=np.array([point]), points=[point+targets, self.intersections], sphere=True)

        # for each input viewpoint, determine and store the difference between the ray angles of the viewpoint and the ray angles of the synthesized point (deviation angles)
        for counter, viewpoint_i in enumerate(indices):
            position = self.capture_set.get_position(viewpoint_i)

            # get the rays from this viewpoint that hit the intersection points
            theta, phi = calc_ray_angles(position, self.intersections)
            rays = calc_uvector_from_angle(theta, phi, self.capture_set.radius)
            # calculate the deviation angles (angle between the two vectors of each point is acos of dot product between rays (viewpoint vectors) and targets (synth point vectors))
            dot = np.sum((rays.flatten() * targets.flatten()).reshape(rays.shape), axis=2)
            dev_angles = np.arccos(np.around(dot, 5))  # round to avoid precision errors leading to values >|1|
            self.dev_angles[:, :, counter] = dev_angles

            #            self.capture_set.draw_scene(indices=[viewpoint_i], s_points=np.array([point]), points=[position + rays, intersections, point+targets], sphere=False, rays=[[position+rays, intersections]])

            # get the uv coordinates that correspond to the rays
            u, v = projections.world2latlong(rays[:, :, 0], rays[:, :, 2], rays[:, :,
                                                                           1])  # switch y and z because EnvironmentMap has a different representation
            self.uvs[viewpoint_i] = np.dstack((u, v))

        out = self.blend_image(knn)
        return out

    def blend_image(self, knn):
        """
        Blends the reprojected images of the k nearest neighbors according to a weighting scheme based on the deviation angles for each pixel
        knn: number of neighbors to incorporate in the result image
        returns: the blended image

        Note: This function uses information stored by the interpolation function and also adds information
        """
        # get the knn first indices and angles of the sorted deviation angles (these indices are in [0,cap_set.get_size()] and are not the actual viewpoint indices)
        best_indices = np.argsort(self.dev_angles, axis=-1)[:, :, :knn]
        self.best_indices = best_indices
        dev_angles = np.sort(self.dev_angles, axis=-1)[:, :,
                     :knn]  # TODO combine argsort and sort? or get values by index

        # use these deviation angles to calculate the weights
        # weights matrix is an "image" of weights in order of best-worst angle
        weights = 1 / (1 + np.exp(500 * (dev_angles - 0.017)))
        #        weights = 1 / (1 + np.exp(4*(np.rad2deg(dev_angles) - 1)))

        # modify the weights so that they sum up to 1
        sums = np.sum(weights, axis=-1)
        weights /= sums[:, :, np.newaxis]

        # find all of the viewpoints that are in the top knn and reproject these viewpoints using the corresponding uv coordinates in self.uvs
        used_indices = np.unique(best_indices)
        # prepare array to store the reprojected images
        reproj_imgs = np.zeros((used_indices.shape[0], self.dev_angles.shape[0], self.dev_angles.shape[1], 3))

        for i, vp_i in enumerate(used_indices):
            reprojected = self.reproject(self.capture_set.get_capture(self.indices[vp_i]).img, self.uvs[
                self.indices[vp_i]])  # use real viewpoint index instead of the location in the used_index list
            viewpoint_indices = np.nonzero(best_indices == vp_i)

            # masks will be filled with eigher weight or 0 depending on how much influence the pixel at each rank should have (rank as in best - k-best deviation angle)
            mask = np.zeros_like(weights)
            mask[viewpoint_indices] = weights[viewpoint_indices]
            mask = np.add.reduce(mask, axis=2)

            # multiply mask by weight for each index used, then multiply by reprojection
            reproj_imgs[i] = reprojected * mask[:, :, np.newaxis]
        #            utils.cvwrite(reproj_imgs[i], "masked_" + str(self.indices[vp_i]) + ".jpg" )

        # sum up all masked & weighted reprojections
        image = np.add.reduce(reproj_imgs, axis=0)
        return image

    def reproject(self, image, uvs):
        """
        reprojects an image using the EnvironmentMap.interpolate function
        """
        envmap = EnvironmentMap(image, "latlong")
        u, v = np.split(uvs, 2, axis=2)
        envmap.interpolate(u, v)
        return envmap.data

    def weight(self, angle_w, dist_w):
        """
        the closer to 0 the better
        """
        distance_vectors = self.capture_set.get_positions(self.indices) - self.point
        distances = np.sqrt(np.sum(np.power(distance_vectors, 2), axis=-1))
        self.distances = distances
        n_distances = distances / (self.capture_set.radius * 2)

        w = 1
        metrics = angle_w * self.dev_angles + dist_w * n_distances

    def show_point_influences(self, v, u, show_inset=True, sphere=True, best_arrows=False):
        """
        Shows the which viewpoints influence the pixel at a certain position (uv) in the last synthesized viewpoint. The pixel at the position uv is translated into world coordinates.
        v: y position of the pixel
        u: x position of the pixel
        show_inset: show the locations and deviation angles of all other viewpoints in the capture set
        sphere: show the scene model (sphere)
        best_arrows: draw arrows from the synthesized point (in orange) and the best knn viewpoints (in blue) to the point P (P = world_coordinates(uv)
        """
        uv_index = np.array([uv_height, uv_width])
        s_point = self.point
        intersection_point = self.intersections[uv_height, uv_width]
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # best indices (actual viewpoint indices) for point uv
        best_real_indices = [self.indices[b_i] for b_i in self.best_indices[uv_index[0], uv_index[1]]]

        # all deviation angles for point uv
        dev_angles = np.rad2deg(self.dev_angles[uv_index[0], uv_index[1]])
        index2dev = {}
        for i in range(len(dev_angles)):
            index2dev[self.indices[i]] = dev_angles[i]

        # show the scene model (sphere)
        if sphere:
            u = np.linspace(0, np.pi, 15)
            v = np.linspace(0, 2 * np.pi, 15)
            x = np.outer(np.sin(u), np.sin(v)) * self.capture_set.radius
            y = np.outer(np.sin(u), np.cos(v)) * self.capture_set.radius
            z = np.outer(np.cos(u), np.ones_like(v)) * self.capture_set.radius
            ax.plot_wireframe(x, y, z, color='0.8')

        # show the synthesized point
        ax.scatter(s_point[0], s_point[1], s_point[2], color='orange')
        ax.text(s_point[0], s_point[1], s_point[2], 's')

        # show the scene intersection of the uv_index
        ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2], color='green')
        ax.text(intersection_point[0], intersection_point[1], intersection_point[2], 'P')
        # show the ray from the s_point to the intersection
        i_point = intersection_point - s_point
        plt.quiver(s_point[0], s_point[1], s_point[2], i_point[0], i_point[1], i_point[2], arrow_length_ratio=0.1,
                   color="orange")

        if best_arrows:
            starts = self.capture_set.get_positions(best_real_indices)
            ends = intersection_point - starts
            plt.quiver(starts[:, 0], starts[:, 1], starts[:, 2], ends[:, 0], ends[:, 1], ends[:, 2],
                       arrow_length_ratio=0.1)

        # show the points used for this uv index
        for i, best_i in enumerate(best_real_indices):
            point = self.capture_set.get_position(best_i)
            ax.scatter(point[0], point[1], point[2], color='lime')
            ax.text(point[0], point[1], point[2], str(np.round(index2dev[best_i], 2)) + "° (" + str(best_i) + ")")

        # show all the other points that are not the best indices
        if show_inset:
            inset_diff = np.setdiff1d(self.indices, best_real_indices, assume_unique=True)
            viewpoints = self.capture_set.get_positions(inset_diff)
            for i, vp in enumerate(viewpoints):
                ax.scatter(vp[0], vp[1], vp[2], color='mediumseagreen')
                # need the deviation angles in order of inset_diff
                # deviation angle argwhere indices == inset_diff[i]
                ax.text(vp[0], vp[1], vp[2],
                        str(np.round(index2dev[inset_diff[i]], 2)) + "° (" + str(inset_diff[i]) + ")")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def visualize_best_deviation(self, saveas=None):
        """
        Saves or shows an image of the best deviation angles to the last synthesized viewpoint.
        """
        dev_angles = np.rad2deg(np.sort(self.dev_angles, axis=-1)[:, :, :1])
        imgplot = plt.imshow(dev_angles, cmap="RdBu_r")
        plt.colorbar()
        if saveas is None:
            plt.show()
        else:
            plt.savefig(saveas)
        plt.clf()

    def visualize_distances(self, saveas=None):
        """
        Saves or shows an image of the distances of the image areas used for synthesis
        """
        distance_vectors = self.capture_set.get_positions(self.indices) - self.point
        distances = np.sqrt(np.sum(np.power(distance_vectors, 2), axis=-1))

        indices = self.best_indices[:, :, 0].flatten()
        distance_patches = distances[indices].reshape(self.dev_angles.shape[:2])
        imgplot = plt.imshow(distance_patches, cmap="RdBu_r")
        plt.colorbar()
        if saveas is None:
            plt.show()
        else:
            plt.savefig(saveas)


#        plt.clf()
#        imgplot = plt.imshow(self.best_indices[:,:,0], cmap="RdBu_r")
#        plt.colorbar()
#        plt.show()
#        plt.clf()

######################### helper functions #########################
def shift_img(img, flow, alpha):
    """
    Shifts an image (in any color space) along the provided flow vectors

    img: the image to be shifted
    flow: the vectors per pixel that img should be shifted by
    alpha: the amount the image should be shifted along the flow vectors

    returns: the shifted image
    """
    xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    xx_shifted = (xx - (flow[:, :, 0] * alpha)).astype(np.float32)
    yy_shifted = (yy - (flow[:, :, 1] * alpha)).astype(np.float32)

    shifted_coords = np.array([yy_shifted.flatten(), xx_shifted.flatten()])
    shifted_img = np.ones_like(img)
    for d in range(img.shape[2]):
        shifted_img[:, :, d] = np.reshape(map_coordinates(img[:, :, d], shifted_coords), (img.shape[0], img.shape[1]))

    return shifted_img


def calc_ray_angles(source, targets):
    """
    Calculates the angles of the rays from a origin point
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions

    source: the origin point of all the rays
    targets: an array of vector end points

    returns: an array of inclinations (theta) and an array of azimuths (phi) of the vector angles
    """
    # move so source is 0,0,0 and normalize
    u_vectors = calc_unit_vectors(targets - source)
    theta = np.arccos(u_vectors[..., 2])  # z
    phi = np.arctan2(u_vectors[..., 1], u_vectors[..., 0])
    return theta, phi


def calc_unit_vectors(vectors):
    """
    Calculates the unit vectors of given vectors
    """
    mag = np.linalg.norm(vectors, axis=2)
    u_vectors = np.zeros_like(vectors)
    u_vectors[:, :, 0] = vectors[:, :, 0] / mag
    u_vectors[:, :, 1] = vectors[:, :, 1] / mag
    u_vectors[:, :, 2] = vectors[:, :, 2] / mag  # TODO find a cleaner way to do this np.dot?
    return u_vectors


def calc_uvector_from_angle(theta, phi, radius=1):
    """
    Calculates the unit vectors from the given inclinations and azimuths
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions

    theta: array of inclinations
    phi: array of azimuths
    radius: TODO this shouldn't be necessary 
    """
    x = np.sin(theta) * np.cos(phi) * radius
    y = np.sin(theta) * np.sin(phi) * radius
    z = np.cos(theta) * radius
    return calc_unit_vectors(np.dstack((x, y, z)))


def calc_uv_coordinates(imgheight):
    """
    adapted from EnvironmentMap.imageCoordinates at https://github.com/soravux/skylibs/blob/master/envmap/environmentmap.py
    """
    cols = np.linspace(0, 1, imgheight * 2 * 2 + 1)
    rows = np.linspace(0, 1, imgheight * 2 + 1)

    cols = cols[1::2]
    rows = rows[1::2]

    return [d.astype('float32') for d in np.meshgrid(cols, rows)]
