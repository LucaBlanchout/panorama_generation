from os import listdir, path, makedirs
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from envmap import EnvironmentMap

import utils
import preproc

class Capture:
    """
    simple object storing the position, rotation and the image data of a capture
    """
    def __init__(self, imgpath, pos, rot):
        self.pos = pos
        self.rot = rot
        self.imgpath = imgpath
        self.img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)

    def store_image(self, path=None):
        """
        stores the image at the imgpath if it does not already exist
        overwrites if it does
        """
        if path is None:
            path = self.imgpath

        utils.cvwrite(self.img, path)
        print("storing at " + path)


    def rotate(self, rotation):
        envmap = EnvironmentMap(self.img, 'latlong')
        envmap.rotate('DCM', rotation.as_matrix())
        self.img = envmap.data

class CaptureSet:
    """
    set with all captures that holds metadata and paths and can retrieve pairs of both based on index
    also contains the model of the scene (as a sphere centered around 0,0,0)
    stores positional coordinates in x, y, z order, x/y being the plane parallel to the ground
    CaptureSet()
    """
    def __init__(self, location, radius=None, in_place=False):
        """
        location target must contain
            - a folder named images containing the images of the capture set in the same order as the metadata
            - a file named metadata.txt containing the metadata (the format of the metadata is described in preproc.parse_metadata)
        in_place: if true, images are normalized in place (i.e. rotated)
        """
        self.location = location
        self.names = sorted(listdir(location + "/images"))
        self.positions = np.zeros((len(self.names), 3))
        self.rotations = np.zeros((len(self.names), 4))

        #try loading previously stored, normalized metadata
        try:
            with open(location + '/positions.npy', 'rb') as f:
                self.positions = np.load(f)

            with open(location + '/rotations.npy', 'rb') as f:
                self.rotations = np.load(f)
        except FileNotFoundError:
            print("No previously stored information found, loading and normalizing metadata")
            if not in_place:
                imgs = location + '/images_raw'
                if not path.exists(imgs):
                    print("no path " + imgs)
                    try:
                        makedirs(imgs)
                    except OSError as exc: # guard agains race condition
                        if exc.ernno != errno.EEXIST:
                            raise

                #copy images over as backup
                for i in range(len(self.names)):
                    print("backing up " + self.names[i])
                    img = cv2.cvtColor(self.get_capture(i).img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(location + '/images_raw/' + str(i) + '.jpg', img)

            #get raw positions and rotations
            raw_pos, raw_rot = preproc.parse_metadata(location + "/metadata.txt")

            #center points
            self.positions = preproc.center(raw_pos)
            self.store_positions()

            self.rotations = raw_rot
            preproc.normalize_rotation(self)

        self.set_scene(radius)

    def set_scene(self, radius):
        #TODO why?? center should always be 0,0,0 anyway
        minima = np.amin(self.positions, axis=0)
        maxima = np.amax(self.positions, axis=0)
        self.center = minima + (maxima-minima) * 0.5
        if radius is not None:
            self.radius = radius
        else:
            self.radius = self.get_radius()
        print("radius: ", self.radius)

    def get_size(self):
        return len(self.positions)

    def get_position(self, index):
        """
        retrieves the position of the capture at the specified index
        """
        return self.positions[index]

    def get_positions(self, indices):
        """
        retrieves the positions of the captures at the specified indices
        returns a 2D array containing these positions in order of argument input
        """
        pos = []
        for i in indices:
            pos.append(self.get_position(i))
        return np.array(pos)

    def get_rotation(self, index):
        """
        retrieves the rotation of the capture at the specified index
        """
        return self.rotations[index]

    def get_capture(self, index):
        """
        retrieves the entire capture at the specified index, containing the image, the position and the rotation
        """
        name = self.location + "/images/" + str(index) + ".jpg"
        return Capture(name, self.get_position(index), self.get_rotation(index))

    def get_captures(self, indices):
        captures = {}
        for i in indices:
            captures[i] = self.get_capture(i)
        return captures

#    def update_captures(self, captures):
#        for k, v in captures.items():
#            self.position[int(k)] = v.pos
#            self.rotation[int(k)] = v.rot
#            v.store_image()

    def store_rotations(self, location=None):
        if location is None:
            location = self.location
        with open(location + '/rotations.npy', 'wb') as f:
            np.save(f, self.rotations)

    def store_positions(self, location=None):
        if location is None:
            location = self.location
        with open(location + '/positions.npy', 'wb') as f:
            np.save(f, self.positions)

#    def store(self, field):
#        if field is "positions":
#            pass
#        elif field is "rotations":
#            pass
#        elif field is "image":
#            pass
#        else:
#

    def get_radius(self):
        """
        gets or calculates the (estimated) radius of the scene
        at the moment this is a placeholder function that returns a radius that is slightly larger than the furthest point but in the end this should return a more accurate scene radius
        """
        buf = 0.1
        maxima = np.amax(np.abs(self.positions), axis=0)
        rad = np.sqrt(np.power(maxima[0], 2) + np.power(maxima[1], 2))
        return (rad) * (1 + buf)

    def calc_ray_intersection(self, point, vectors):
        """
        calculates the points at which the rays (point-vectors) intersect the scene
        https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
        http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/

        point is origin, t is distance and D is unit vectors centered around 0,0,0
        x^2 + y^2 + z^2 = R^2 sphere function
        P^2 - R^2 = 0
        ...
        O^2 + D^2t^2 + 2ODt - R^O
        quadratic function with a=D^2, b= 2OD, c=O^2-R^2

        """
        #dot product of a unit vector with itself is 1
        a = np.ones(vectors.shape[:2])

        f_vecs = vectors.flatten()
        f_points = np.full((vectors.shape[0]*vectors.shape[1],vectors.shape[2]), point).flatten()
        dot = np.sum((f_vecs * f_points).reshape(vectors.shape), axis=2)
        b = 2 * dot
        c = np.full_like(a, np.dot(point, point)- np.power(self.radius, 2)) 

        discriminants = np.power(b, 2) - (4 * a * c)
        if np.amin(discriminants) < 0:
            #TODO specify and throw error
            print("no intersection found at some point")
            return
        t1 = (-b + np.sqrt(discriminants)) / (2 * a)
        t2 = (-b - np.sqrt(discriminants)) / (2 * a)
        #select the points with positive lengths
        lengths = t1 if np.amin(t1) >= 0 else t2 #TODO make selection for each point separately
        intersections = point + (vectors * lengths[:,:,np.newaxis])
        
        return intersections

    def draw_scene(self, indices=None, s_points=None, sphere=True, points=None, rays=None, numpoints=200):
        """
        draws the scene as a set of points and the containing sphere representing the scene
        indices=None -> all points are drawn
        indices=[a,b,c] only points at indices a,b,c are drawn
        s_points -> extra points (i.e. synthesized points) are drawn in a separate color
        if sphere=False, the sphere representing the scene is omitted
        points is a list of point arrays that can be drawn (e.g. intersections)
        this is a member function instead of a free function, so it can ensure correct handling of the axes (may change this later)
        example usage:
        capture_set.draw_scene(indices=[1,4], s_points=np.array([point]), points=[position + rays, intersections, point+targets], sphere=False, rays=[[position+rays, intersections]])
        """
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        #draw captured viewpoints
        if indices is not None:
            viewpoints = self.positions[[tuple(indices)]]
        else:
            viewpoints = self.positions
            indices = list(range(self.get_size()))
        ax.scatter(viewpoints[:,0], viewpoints[:,1], viewpoints[:,2], color='blue')
        for i in range(len(indices)):
            ax.text(viewpoints[i,0], viewpoints[i,1], viewpoints[i,2], indices[i])

        if points is not None:
            colors = ['green', 'purple', 'magenta', 'cyan', 'yellow']
            for i in range(len(points)):
                p = utils.sample_points(points[i], numpoints)
                ax.scatter(p[:,:,0], p[:,:,1], p[:,:,2], color=colors[i%len(colors)])

        if sphere:
            u = np.linspace(0, np.pi, 15)
            v = np.linspace(0, 2 * np.pi, 15)

            x = np.outer(np.sin(u), np.sin(v)) * self.radius
            y = np.outer(np.sin(u), np.cos(v)) * self.radius
            z = np.outer(np.cos(u), np.ones_like(v)) * self.radius

            ax.plot_wireframe(x, y, z, color='0.8')

        #draw additional (synthesized) points
        if s_points is not None:
            ax.scatter(s_points[:,0], s_points[:,1], s_points[:,2], color='orange')

        if rays is not None:
            for rayset in rays:
                origins = rayset[0]
                targets = rayset[1]
                if origins.shape != (3,):
                    o = utils.sample_points(origins, numpoints)
                else:
                    o = origins
                t = utils.sample_points(targets, numpoints)
                t = t.reshape(-1, t.shape[-1])
                if origins.shape != targets.shape:
                    if origins.shape[0] == 3 or origins.shape[0] == 2:
                        o = np.full_like(t, o)
                    else:
                        raise NotImplementedError
                else:
                    o = o.reshape(-1, o.shape[-1])
                diff = t - o

                plt.quiver(o[:,0], o[:,1], o[:,2], diff[:,0], diff[:,1], diff[:,2], arrow_length_ratio=0.1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
