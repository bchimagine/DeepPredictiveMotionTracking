import numpy as np
import os

from utils.constants import NB_CHANNELS, NB_ANGLES, OUR_MODEL_IMAGE_HEIGHT, IMAGE_HEIGHT, IMAGE_WIDTH, \
    OUR_MODEL_IMAGE_WIDTH, NB_SLICE
from utils.img import (generate_img_angle, whitening,
                       get_rotational_spline_generators)
import SimpleITK as sitk
from tqdm import tqdm
import cv2 as cv


class DataFactory:

    def __init__(self, config):
        self.train_data_path = config["data"]["train_data_path"]
        self.test_data_path = config["data"]["test_data_path"]

        files = os.listdir(config["data"]["train_data_path"])
        self.train_files, self.validation_files = files[:30], files[30:]

        self.test_files = os.listdir(config["data"]["test_data_path"])
        self.est_nb_timesteps = config["data"]["estimation_timestep_limit"]
        self.pred_nb_timesteps = config["data"]["prediction_timestep_limit"]
        self.single_head = config["network"]["single_head"]
        self.mask = config["data"]["mask"]

        if config["network"]["type"] == "our_model" or config["network"]["type"] == "direct_lstm":
            self.height, self.width = OUR_MODEL_IMAGE_HEIGHT, OUR_MODEL_IMAGE_WIDTH
        else:
            self.height, self.width = IMAGE_HEIGHT, IMAGE_WIDTH

        self.resnet18 = True if config["network"]["type"] == "resnet18" else False

        if not config["data"]["initial_slice_index"]:
            self.initial_slice_index = config["data"]["initial_slice_index"]
        else:
            self.initial_slice_index = 55

        if not config["data"]["initial_slice_index"]:
            self.inter_slice_spacing = config["data"]["inter_slice_spacing"]
        else:
            self.inter_slice_spacing = 5

    def generate_data(self, path, files=None):
        """generates data containing batch_size samples"""
        speeds = [4] * 4 + [5] * 5 + [6] * 6 + [7] * 8 + [8] * 9  # 32 samples total == batch_size

        nb_samples = len(speeds)
        nb_files = len(files)
        imgs = np.zeros([nb_files * nb_samples, self.est_nb_timesteps, self.height, self.width, NB_CHANNELS])
        rotational = np.zeros(
            [nb_files * nb_samples, self.est_nb_timesteps + self.pred_nb_timesteps, NB_ANGLES])
        offset = np.zeros(
            [nb_files * nb_samples, self.est_nb_timesteps + self.pred_nb_timesteps, NB_CHANNELS])

        for counter, file_name in enumerate(tqdm(files)):
            image = sitk.ReadImage(os.path.join(path, file_name))
            for i in range(nb_samples):
                ius_x, ius_y, ius_z, rotation_matrix_init = get_rotational_spline_generators(
                    self.est_nb_timesteps + self.pred_nb_timesteps, speed=speeds[i])
                timestep_to_mask = np.random.randint(low=0, high=self.est_nb_timesteps, size=1)[
                    0] if self.mask else self.est_nb_timesteps + 1

                for timestep in range(self.est_nb_timesteps + self.pred_nb_timesteps):
                    z = np.random.randint(0, self.inter_slice_spacing, 1)[0]
                    z += self.initial_slice_index + (timestep * self.inter_slice_spacing)
                    img, angle_offsets = generate_img_angle(timestep,
                                                            image,
                                                            ius_x, ius_y, ius_z,
                                                            rotation_matrix_init)
                    rotational[nb_samples * counter + i, timestep, :] = angle_offsets

                    offset[nb_samples * counter + i, timestep, 0] = z
                    if timestep < self.est_nb_timesteps:
                        if self.height != IMAGE_HEIGHT and self.width != IMAGE_WIDTH:
                            img = cv.resize(img, dsize=(self.height, self.width),
                                            interpolation=cv.INTER_NEAREST)

                        # randomly mask a timestep with all black image
                        if timestep == timestep_to_mask:
                            img = np.zeros((self.height, self.width))

                        imgs[nb_samples * counter + i, timestep, :, :, 0] = whitening(img)

        total_records = imgs.shape[0]

        encoder_input = np.zeros((total_records, self.est_nb_timesteps, imgs.shape[2], imgs.shape[3], imgs.shape[4]))
        decoder_input = np.zeros((total_records, self.pred_nb_timesteps, NB_ANGLES + NB_SLICE))
        encoder_estimation_offset_z = np.zeros((total_records, self.est_nb_timesteps, 1))
        encoder_estimation_rotation_xy = np.zeros((total_records, self.est_nb_timesteps, 2))
        encoder_estimation_rotation_z = np.zeros((total_records, self.est_nb_timesteps, 1))
        decoder_prediction_offset_z = np.zeros((total_records, self.pred_nb_timesteps, 1))
        decoder_prediction_rotation_xy = np.zeros((total_records, self.pred_nb_timesteps, 2))
        decoder_prediction_rotation_z = np.zeros((total_records, self.pred_nb_timesteps, 1))
        for i in range(total_records):
            encoder_input[i, :self.est_nb_timesteps] = imgs[i, :self.est_nb_timesteps]

            encoder_estimation_rotation_xy[i] = rotational[i, :self.est_nb_timesteps, :2]
            decoder_prediction_rotation_xy[i] = rotational[i, self.est_nb_timesteps:, :2]
            encoder_estimation_rotation_z[i] = rotational[i, :self.est_nb_timesteps, 2:]
            decoder_prediction_rotation_z[i] = rotational[i, self.est_nb_timesteps:, 2:]
            encoder_estimation_offset_z[i] = offset[i, :self.est_nb_timesteps]
            decoder_prediction_offset_z[i] = offset[i, self.est_nb_timesteps:]

        x = {"encoder_input": encoder_input, "decoder_input": decoder_input}

        if self.single_head:
            if self.resnet18:
                x = {"encoder_input": encoder_input.reshape((total_records * self.est_nb_timesteps, imgs.shape[2],
                                                             imgs.shape[3], imgs.shape[4])),
                     "decoder_input": decoder_input}

                y = {"estimation_offset_z": encoder_estimation_offset_z.reshape((total_records * self.est_nb_timesteps,
                                                                                 1)),
                     "estimation_rotation": np.concatenate((encoder_estimation_rotation_xy,
                                                            encoder_estimation_rotation_z),
                                                           axis=2).reshape((total_records * self.est_nb_timesteps,
                                                                            NB_ANGLES)),
                     "prediction_offset_z": decoder_prediction_offset_z.reshape((total_records *
                                                                                 self.pred_nb_timesteps, 1)),
                     "prediction_rotation": np.concatenate((decoder_prediction_rotation_xy,
                                                            decoder_prediction_rotation_z),
                                                           axis=2).reshape((total_records * self.pred_nb_timesteps, 3))}
            else:
                y = {"estimation_offset_z": encoder_estimation_offset_z,
                     "estimation_rotation": np.concatenate((encoder_estimation_rotation_xy,
                                                            encoder_estimation_rotation_z),
                                                           axis=2),
                     "prediction_offset_z": decoder_prediction_offset_z,
                     "prediction_rotation": np.concatenate((decoder_prediction_rotation_xy,
                                                            decoder_prediction_rotation_z),
                                                           axis=2)}
        else:
            y = {"estimation_offset_z": encoder_estimation_offset_z,
                 "estimation_rotation_xy": encoder_estimation_rotation_xy,
                 "estimation_rotation_z": encoder_estimation_rotation_z,
                 "prediction_offset_z": decoder_prediction_offset_z,
                 "prediction_rotation_xy": decoder_prediction_rotation_xy,
                 "prediction_rotation_z": decoder_prediction_rotation_z}

        return x, y

    def generate_train_data(self):
        return self.generate_data(self.train_data_path, self.train_files[:1])

    def generate_validation_data(self):
        return self.generate_data(self.train_data_path, self.validation_files[:1])

    def generate_test_data(self):
        return self.generate_data(self.test_data_path, self.test_files[:1])
