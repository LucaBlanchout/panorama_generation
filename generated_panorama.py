import utils
import optical_flow
from skylibs.envmap import EnvironmentMap
from cubemap import ExtendedCubeMap

import numpy as np
import itertools
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


class GeneratedPanorama:
    def __init__(self, rho, side, shape, camera_container, cameras_to_keep, envmap_type, base_out_path):
        self.rho = rho
        self.side = side
        self.data = np.zeros(shape).reshape((-1, 3))
        self.shape = shape

        self.camera_container = camera_container
        self.cameras_to_keep = cameras_to_keep
        self.cameras_to_keep_range = tuple(range(1, cameras_to_keep + 1))

        self.envmap_type = envmap_type
        self.base_out_path = base_out_path

        self.eye_points = None
        self.eye_vectors = None

        self.angles_eyes_cameras = None
        self.min_angles = None
        self.min_angles_indexes = None
        self.min_angles_ratio = None
        self.intermediate_points = None
        self.best_cameras_vectors = None

    def calculate_eye_vectors(self, projection_points):
        vx = projection_points[:, 0] - self.eye_points[:, 0]
        vy = projection_points[:, 1] - self.eye_points[:, 1]
        vz = projection_points[:, 2] - self.eye_points[:, 2]

        magnitude = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

        vx /= magnitude
        vy /= magnitude
        vz /= magnitude

        eye_vectors = np.stack((vx, vy, vz), axis=1)

        self.eye_vectors = eye_vectors

    def calculate_angles_eyes_cameras(self):
        def dot_product(vector1, vector2):
            return np.sum((vector1 * vector2), axis=1)

        def length(v):
            return np.sqrt(dot_product(v, v))

        angles = []
        for camera_vectors in self.camera_container.get_cameras_vectors_cartesian():
            ang_rad = np.arccos(dot_product(self.eye_vectors, camera_vectors) / length(self.eye_vectors) * length(camera_vectors))

            ang_deg = np.degrees(ang_rad) % 360

            ang_deg = np.where(ang_deg - 180 >= 0, 360 - ang_deg, ang_deg)

            angles.append(ang_deg)

        self.angles_eyes_cameras = np.transpose(np.array(angles))
        self.calculate_minimum_angles_and_indexes_and_ratio()

    def calculate_minimum_angles_and_indexes_and_ratio(self):
        self.min_angles_indexes = np.argpartition(self.angles_eyes_cameras, self.cameras_to_keep_range)[:, :self.cameras_to_keep_range[-1]]
        self.min_angles = self.angles_eyes_cameras[np.arange(self.angles_eyes_cameras.shape[0])[:, None], self.min_angles_indexes]
        min_angles_ratio = self.min_angles[:, 0] / (self.min_angles[:, 0] + self.min_angles[:, 1])
        min_angles_ratio_inv = 1 - min_angles_ratio
        self.min_angles_ratio = np.column_stack((min_angles_ratio, min_angles_ratio_inv))

    def calculate_intermediate_points(self):
        self.intermediate_points = np.empty((self.min_angles_ratio.shape[0], 3))

        cameras_coordinates = self.camera_container.get_cameras_coordinates()

        camera_permutation_indices = list(itertools.permutations(range(len(cameras_coordinates)), 2))

        for permutation_index in camera_permutation_indices:
            permutation_index = np.array(permutation_index)
            indices = np.argwhere((self.min_angles_indexes == permutation_index).all(axis=1))

            alphas = self.min_angles_ratio[indices]
            inv_alphas = 1 - self.min_angles_ratio[indices]

            xs = alphas * cameras_coordinates[permutation_index[0]][0] + inv_alphas * cameras_coordinates[permutation_index[1]][0]
            ys = alphas * cameras_coordinates[permutation_index[0]][1] + inv_alphas * cameras_coordinates[permutation_index[1]][1]
            zs = alphas * cameras_coordinates[permutation_index[0]][2] + inv_alphas * cameras_coordinates[permutation_index[1]][2]

            self.intermediate_points[indices, 0] = xs
            self.intermediate_points[indices, 1] = ys
            self.intermediate_points[indices, 2] = zs

    def calculate_best_cameras_vectors(self):
        cameras_vectors = self.camera_container.get_cameras_vectors_cartesian()

        self.best_cameras_vectors = cameras_vectors[
                self.min_angles_indexes,
                np.arange(cameras_vectors.shape[1])[:, None],
                :
            ]

    def generate_panorama(self, base_panorama_container):
        height, width, channels = self.shape
        self.data = np.zeros(self.shape).reshape((-1, 3))

        for i in range(len(base_panorama_container)):
            for j in range(self.cameras_to_keep):

                index = np.argwhere(self.min_angles_indexes[:, j] == i)

                xyz = np.squeeze(self.best_cameras_vectors[index, j, :])

                u, v = base_panorama_container[i].envmap.world2image(xyz[:, 0], xyz[:, 1], xyz[:, 2])

                u = (u * width).astype(np.int)
                v = (v * height).astype(np.int)

                u = np.where(u >= width, width - 1, u)
                v = np.where(v >= height, height - 1, v)

                self.data[index, :] += base_panorama_container[i].envmap.data[v, u, :][:, None]

        self.data = (self.data.reshape(self.shape) * 255 / self.cameras_to_keep).astype(np.uint8)

        self.write_data()

    def generate_panorama_temp(self, base_panorama_container, interpolated_base_panoramas):
        print('temp1')

        height, width, channels = self.shape
        self.data = np.zeros(self.shape).reshape((-1, 3))

        for i in range(len(base_panorama_container)):
            for j in range(self.cameras_to_keep):
                min_angles_index = np.argwhere(self.min_angles_indexes[:, j] == i)

                min_angles_ratio = self.min_angles_ratio[min_angles_index, j]

                xyz = np.squeeze(self.best_cameras_vectors[min_angles_index, j, :])

                u, v = base_panorama_container[i].envmap.world2image(xyz[:, 0], xyz[:, 1], xyz[:, 2])

                u = (u * width).astype(np.int)
                v = (v * height).astype(np.int)

                u = np.where(u >= width, width - 1, u)
                v = np.where(v >= height, height - 1, v)
                print('xyz =', xyz.shape)
                print('u =', u.shape)
                print('v =', v.shape)
                print('angles index =', min_angles_index.shape)
                print('angles ratio=', min_angles_ratio.shape)
                print('')

                # self.data[min_angles_index, :] += np.expand_dims(min_angles_ratio * base_panorama_container[i].envmap.data[v, u, :], axis=1)

        # self.data = (self.data.reshape(self.shape) * 255).astype(np.uint8)
        #
        # self.write_data()

    def generate_panorama_temp_2(self, base_panorama_container, interpolated_base_panoramas):
        height, width, channels = self.shape
        self.data = np.zeros(self.shape).reshape((-1, 3))

        for pair in interpolated_base_panoramas.keys():
            pair_array = np.array(pair)
            min_angles_index = np.argwhere(np.all(self.min_angles_indexes == pair_array, axis=1))

            min_angles_ratio = np.squeeze(self.min_angles_ratio[min_angles_index])

            min_angles_ratio = np.around(min_angles_ratio, decimals=2)

            print('angles index =', min_angles_index.shape)
            print('angles ratio=', min_angles_ratio.shape)

            # for i in range(len(pair_array)):
            for i, index in enumerate(pair_array):
                xyz = np.squeeze(self.best_cameras_vectors[min_angles_index, i, :])

                u, v = base_panorama_container[index].envmap.world2image(xyz[:, 0], xyz[:, 1], xyz[:, 2])

                u = (u * width).astype(np.int)
                v = (v * height).astype(np.int)

                u = np.where(u >= width, width - 1, u)
                v = np.where(v >= height, height - 1, v)

                for j in range(u.shape[0]):
                    try:
                        self.data[min_angles_index[j], :] += [min_angles_ratio[j, 0]] * interpolated_base_panoramas[pair][min_angles_ratio[j, 0]][i][v[j], u[j], :]
                    except (KeyError, IndexError) as e:
                        pass

                # plt.imshow((self.data.reshape(self.shape) / 2).astype(np.uint8))
                # plt.show()

        self.data = (self.data.reshape(self.shape)).astype(np.uint8)
        plt.imshow(self.data)
        plt.show()

        self.write_data()

    def generate_panorama_temp_3(self, base_panorama_container, interpolated_base_panoramas):
        height, width, channels = self.shape
        self.data = np.zeros(self.shape).reshape((-1, 3))

        for pair in interpolated_base_panoramas.keys():
            pair_array = np.array(pair)
            min_angles_index = np.argwhere(np.all(self.min_angles_indexes == pair_array, axis=1))

            min_angles_ratio = np.squeeze(self.min_angles_ratio[min_angles_index])

            min_angles_ratio = np.around(min_angles_ratio, decimals=2)

            xyz = np.squeeze(self.best_cameras_vectors[min_angles_index, 0, :])

            u, v = base_panorama_container[0].envmap.world2image(xyz[:, 0], xyz[:, 1], xyz[:, 2])

            u = (u * width).astype(np.int)
            v = (v * height).astype(np.int)

            u = np.where(u >= width, width - 1, u)
            v = np.where(v >= height, height - 1, v)

            for j in range(u.shape[0]):
                try:
                    self.data[min_angles_index[j], :] = interpolated_base_panoramas[pair][min_angles_ratio[j, 0]][v[j], u[j], :]
                except (KeyError, IndexError):
                    pass

        self.data = (self.data.reshape(self.shape)).astype(np.uint8)

        self.write_data()

    def write_data(self):
        out_path_latlong = self.base_out_path + 'latlong/'
        Path(out_path_latlong).mkdir(parents=True, exist_ok=True)

        if self.envmap_type == 'latlong':
            panorama_latlong = Image.fromarray(self.data)
            save_path = out_path_latlong + "rho_" + str(self.rho) + '_' + self.side + '.jpg'
            panorama_latlong.save(save_path)
            print("Saved latlong representation in :", save_path)

        elif self.envmap_type == 'cube':
            panorama_cube = Image.fromarray(self.data)
            out_path_cube = self.base_out_path + 'cubemap_ratio/'
            Path(out_path_cube).mkdir(parents=True, exist_ok=True)
            save_path = out_path_cube + "cubemap_rho_" + str(self.rho) + '_' + self.side + '.jpg'
            panorama_cube.save(save_path)
            print("Saved cubemap representation in :", save_path)

            panorama_latlong = EnvironmentMap(self.data, 'cube')
            panorama_latlong = panorama_latlong.convertTo('latlong')
            panorama_latlong = Image.fromarray(panorama_latlong.data.astype(np.uint8))
            save_path = out_path_latlong + "rho_" + str(self.rho) + '_' + self.side + '.jpg'
            panorama_latlong.save(save_path)
            print("Saved latlong representation in :", save_path)


class GeneratedPanoramaContainer:
    def __init__(self, base_panorama_container, camera_container, cameras_to_keep, envmap_type='latlong', base_out_path='out/', rho=1.0, viewing_circle_radius=0.032):
        self.side = ['left', 'right']
        self.generated_panoramas_dict = {}
        self.rho = rho
        self.base_panorama_container = base_panorama_container
        self.camera_container = camera_container
        self.base_out_path = base_out_path

        for side in self.side:
            self.generated_panoramas_dict[side] = GeneratedPanorama(
                rho,
                side,
                self.base_panorama_container.base_panorama_shape,
                self.camera_container,
                cameras_to_keep,
                envmap_type,
                self.base_out_path
            )

        self.viewing_circle_radius = viewing_circle_radius
        self.projection_points = None
        self.cameras_vectors_cartesian = None
        self.cameras_vectors_spherical = None

        self.optical_flows = dict()

        # TODO : These should be temporary
        self.interpolated_base_panoramas = dict()

        self.calculate_optical_flows_permutations()
        # self.calculate_min_max_ratio_for_interpolation()
        self.calculate_min_max_ratio_for_interpolation_2()

    def __getitem__(self, item):
        return self.generated_panoramas_dict[item]

    def __iter__(self):
        yield from self.generated_panoramas_dict.items()

    def generate_panoramas(self):
        self.calculate_projection_points()
        self.calculate_eye_points()
        self.calculate_camera_vectors()
        self.calculate_eye_vectors()
        self.calculate_angles_between_eyes_and_cameras_vectors()
        self.calculate_best_cameras_vectors()


        for side, generated_panorama in self.generated_panoramas_dict.items():
            # generated_panorama.generate_panorama_temp(self.base_panorama_container, self.interpolated_base_panoramas)
            # generated_panorama.generate_panorama_temp_2(self.base_panorama_container, self.interpolated_base_panoramas)
            generated_panorama.generate_panorama_temp_3(self.base_panorama_container, self.interpolated_base_panoramas)

    def set_rho_and_generate_panoramas(self, rho):
        self.rho = rho

        for side, generated_panorama in self:
            generated_panorama.rho = rho

        self.generate_panoramas()

    def calculate_projection_points(self):
        x, y, z, valid = self.base_panorama_container.get_world_coordinates()

        xs = self.rho * x
        ys = self.rho * y
        zs = self.rho * z

        self.projection_points = np.stack((xs, ys, zs), axis=2).reshape(-1, 3)

    def calculate_eye_points(self):
        b = np.sqrt(self.projection_points[:, 0] ** 2 + self.projection_points[:, 2] ** 2)

        viewing_circle_radius_array = np.empty(b.shape)
        viewing_circle_radius_array.fill(self.viewing_circle_radius)

        th = np.arccos(np.divide(viewing_circle_radius_array, b))

        ds = np.arctan2(self.projection_points[:, 2], self.projection_points[:, 0])

        d1 = ds - th
        d2 = ds + th

        ds = [d1, d2]

        for side, d in zip(self.side, ds):
            xs = self.viewing_circle_radius * np.cos(d)
            ys = np.zeros(xs.shape)
            zs = self.viewing_circle_radius * np.sin(d)

            eye_points = np.stack((xs, ys, zs), axis=1)

            eye_points = np.nan_to_num(eye_points)

            self.generated_panoramas_dict[side].eye_points = eye_points

    def calculate_camera_vectors(self):
        self.camera_container.calculate_cameras_vectors(self.projection_points)

    def calculate_eye_vectors(self):
        for side, generated_panorama in self:
            generated_panorama.calculate_eye_vectors(self.projection_points)

    def calculate_angles_between_eyes_and_cameras_vectors(self):
        for side, generated_panorama in self:
            generated_panorama.calculate_angles_eyes_cameras()

    def calculate_best_cameras_vectors(self):
        for side, generated_panorama in self:
            generated_panorama.calculate_best_cameras_vectors()

    def calculate_optical_flows_combinations(self):
        base_panorama_combination_indexes = list(itertools.combinations(range(len(self.base_panorama_container)), 2))

        for combination_index in base_panorama_combination_indexes:
            base_pano_1 = self.base_panorama_container[combination_index[0]]
            base_pano_2 = self.base_panorama_container[combination_index[1]]

            opt_flow = optical_flow.OpticalFlow(
                base_pano_1.extended_cubemap,
                base_pano_2.extended_cubemap,
                combination_index,
                self.base_out_path + 'flow/'
            )

            self.optical_flows[combination_index] = opt_flow

    def calculate_optical_flows_permutations(self):
        base_panorama_permutation_indexes = list(itertools.permutations(range(len(self.base_panorama_container)), 2))

        for permutation_index in base_panorama_permutation_indexes:
            base_pano_1 = self.base_panorama_container[permutation_index[0]]
            base_pano_2 = self.base_panorama_container[permutation_index[1]]

            opt_flow = optical_flow.OpticalFlow(
                base_pano_1.extended_cubemap,
                base_pano_2.extended_cubemap,
                permutation_index,
                self.base_out_path + 'flow/'
            )

            self.optical_flows[permutation_index] = opt_flow

    # def interpolate_base_panoramas(self):
    #     for panorama_pair_index, flow in self.optical_flows.items():
    #         interpolation_path = self.base_out_path + 'flow/interpolation/' + str(panorama_pair_index[0]) + '_to_' + str(
    #                 panorama_pair_index[1]) + '/'
    #         Path(interpolation_path).mkdir(parents=True, exist_ok=True)
    #
    #         for alpha in np.around(np.linspace(0, 1, 11), 1):
    #             base_pano_1 = self.base_panorama_container[panorama_pair_index[0]]
    #             base_pano_2 = self.base_panorama_container[panorama_pair_index[1]]
    #
    #             split_flow = utils.split_cube(flow.flows[0])
    #             shift_1 = {}
    #             for face in utils.FACES:
    #                 shift_1[face] = utils.shift_img(base_pano_1.extended_cubemap.extended[face], split_flow[face], alpha)
    #             shift_1 = utils.build_cube(shift_1)
    #
    #             split_flow = utils.split_cube(flow.flows[1])
    #             shift_2 = {}
    #             for face in utils.FACES:
    #                 shift_2[face] = utils.shift_img(base_pano_2.extended_cubemap.extended[face], split_flow[face],
    #                                                 1 - alpha)
    #             shift_2 = utils.build_cube(shift_2)
    #
    #             out = (1 - alpha) * shift_1 + alpha * shift_2
    #
    #             cv2.imwrite(interpolation_path + 'interpolated_' + str(alpha) + '.jpg', cv2.cvtColor(np.float32(out), cv2.COLOR_RGB2BGR))
    #
    #             out_extended_cube_map = ExtendedCubeMap(out, "Xcube", fov=base_pano_1.extended_cubemap.fov, w_original=base_pano_1.extended_cubemap.w_original)
    #
    #             out_clipped = out_extended_cube_map.get_clipped_cube()
    #
    #             cv2.imwrite(interpolation_path + 'clip_interpolated_' + str(alpha) + '.jpg', cv2.cvtColor(np.float32(out_clipped), cv2.COLOR_RGB2BGR))

    def calculate_min_max_ratio_for_interpolation(self):
        print("Starting interpolation")

        min_ratio = np.nanmin(self['left'].min_angles_ratio[:, 0])
        max_ratio = np.nanmax(self['left'].min_angles_ratio[:, 0])

        min_ratio = round(min_ratio, 2)
        max_ratio = round(max_ratio, 2)

        interpolation_to_generate = np.arange(min_ratio, max_ratio + 0.01, 0.01)

        for panorama_pair_index, flow in self.optical_flows.items():
            print("Starting interpolation for pair :", panorama_pair_index)
            pair_dict = {}
            for ratio in interpolation_to_generate:
                ratio = round(ratio, 2)

                base_pano_1 = self.base_panorama_container[panorama_pair_index[0]]
                base_pano_2 = self.base_panorama_container[panorama_pair_index[1]]

                split_flow = utils.split_cube(flow.flows[0])
                shift_1 = {}
                for face in utils.FACES:
                    shift_1[face] = utils.shift_img(base_pano_1.extended_cubemap.extended[face], split_flow[face],
                                                    ratio)
                shift_1 = utils.build_cube(shift_1)

                shift_1 = ExtendedCubeMap(shift_1, "Xcube", fov=base_pano_1.extended_cubemap.fov, w_original=base_pano_1.extended_cubemap.w_original)
                shift_1 = shift_1.get_clipped_cube()

                split_flow = utils.split_cube(flow.flows[1])
                shift_2 = {}
                for face in utils.FACES:
                    shift_2[face] = utils.shift_img(base_pano_2.extended_cubemap.extended[face], split_flow[face],
                                                    1 - ratio)
                shift_2 = utils.build_cube(shift_2)

                shift_2 = ExtendedCubeMap(shift_2, "Xcube", fov=base_pano_1.extended_cubemap.fov,
                                          w_original=base_pano_1.extended_cubemap.w_original)

                shift_2 = shift_2.get_clipped_cube()

                pair_dict[ratio] = [shift_1, shift_2]

            self.interpolated_base_panoramas[panorama_pair_index] = pair_dict

    def calculate_min_max_ratio_for_interpolation_2(self):
        print("Starting interpolation")

        # min_ratio = np.nanmin(self['left'].min_angles_ratio[:, 0])
        # max_ratio = np.nanmax(self['left'].min_angles_ratio[:, 0])

        # min_ratio = round(min_ratio, 2)
        # max_ratio = round(max_ratio, 2)

        min_ratio = 0.01
        max_ratio = 0.50

        interpolation_to_generate = np.arange(min_ratio, max_ratio + 0.01, 0.01)

        for panorama_pair_index, flow in self.optical_flows.items():
            print("Starting interpolation for pair :", panorama_pair_index)
            pair_dict = {}
            for ratio in interpolation_to_generate:
                ratio = round(ratio, 2)

                base_pano_1 = self.base_panorama_container[panorama_pair_index[0]]
                base_pano_2 = self.base_panorama_container[panorama_pair_index[1]]

                split_flow = utils.split_cube(flow.flows[0])
                shift_1 = {}
                for face in utils.FACES:
                    shift_1[face] = utils.shift_img(base_pano_1.extended_cubemap.extended[face], split_flow[face],
                                                    ratio)
                shift_1 = utils.build_cube(shift_1)

                split_flow = utils.split_cube(flow.flows[1])
                shift_2 = {}
                for face in utils.FACES:
                    shift_2[face] = utils.shift_img(base_pano_2.extended_cubemap.extended[face], split_flow[face],
                                                    1 - ratio)
                shift_2 = utils.build_cube(shift_2)

                out = (1 - ratio) * shift_1 + ratio * shift_2

                out_extended_cube_map = ExtendedCubeMap(out, "Xcube", fov=base_pano_1.extended_cubemap.fov,
                                                        w_original=base_pano_1.extended_cubemap.w_original)

                out_clipped = out_extended_cube_map.get_clipped_cube()

                pair_dict[ratio] = out_clipped

            self.interpolated_base_panoramas[panorama_pair_index] = pair_dict
