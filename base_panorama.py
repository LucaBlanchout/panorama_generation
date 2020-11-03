from skylibs.envmap import EnvironmentMap
from cubemap import ExtendedCubeMap
import cv2
import numpy as np


class BasePanorama:
    def __init__(self, index, in_path, base_out_path='out/', envmap_type='latlong'):
        self.index = index
        self.envmap = EnvironmentMap(in_path + str(self.index) + '.jpg', 'latlong')
        self.extended_cubemap = ExtendedCubeMap((self.envmap.data * 255).astype(np.uint8), 'latlong')
        self.type = envmap_type
        self.base_out_path = base_out_path

        if self.type == 'cube':
            self.envmap = self.envmap.convertTo('cube')

        self.bgr_img = cv2.cvtColor((self.envmap.data * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        self.grey_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2GRAY)

        self.bgr_extended_cubemap = cv2.cvtColor(np.float32(self.extended_cubemap.get_extended_cube()), cv2.COLOR_RGB2BGR)
        self.grey_extended_cubemap = cv2.cvtColor(self.bgr_extended_cubemap, cv2.COLOR_BGR2GRAY)

        self.shape = self.bgr_img.shape

    def write_pano(self):
        cv2.imwrite(self.base_out_path + 'base_' + str(self.index) + '.jpg', self.bgr_img)


class BasePanoramaContainer:
    def __init__(self, base_panoramas=None, base_out_path='out/'):
        self.base_panoramas = base_panoramas if base_panoramas is not None else []
        self.base_out_path = base_out_path

    def __len__(self):
        return len(self.base_panoramas)

    def __getitem__(self, item):
        return self.base_panoramas[item]

    def __iter__(self):
        yield from self.base_panoramas

    def append(self, base_panorama):
        self.base_panoramas.append(base_panorama)

    @property
    def base_panorama_shape(self):
        return self[0].shape

    def get_world_coordinates(self):
        x, y, z, valid = self[0].envmap.worldCoordinates()
        return x, y, z, valid

    def write_base_panoramas(self):
        for base_panorama in self:
            base_panorama.write_pano()
