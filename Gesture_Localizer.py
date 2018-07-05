import os
import numpy as np
import tensorflow as tf
import pandas as pd

from keras import backend as K
from keras.layers import Input, Lambda, ConvLSTM2D, Reshape, Conv2D, MaxPooling2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.optimizers import  Nadam
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization

#getting necessary values from local files
from data_processing import DataGenerator, process_data, get_detector_mask, get_anchors, get_classes
from yolo_body import (preprocess_true_boxes, yolo_body, yolo_eval, yolo_head, yolo_loss)
from visualisation import PlotLearning, draw_boxes

#getting defaults
from parameters.default_values import IMAGESIZE, RESOLUTION, RESTORE_PATHS, DATAPATH, N_CLASSES, YOLO_ANCHORS

#setting up plotting class
plot_losses = PlotLearning()

#defining class which we will use for creating/manipulating model
class Gesture_Localizer():

    def __init__(self, n_classes, batch_size, timestep, lr = 0.0001, dc = 0.004, save_weights = 0, stateful = False):
        self.RESOLUTION = RESOLUTION
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.timestep = timestep
        self.stateful = stateful

        optimizer = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=dc)
        [yolo_model_first, self.model_first] = self.create_first_stage()
        self.model_first.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}, metrics=['accuracy'])

        if save_weights:
            self.model_first.load_weights(RESTORE_PATHS[0])

        self.training_generator, self.validation_generator = self.extract_data(DATAPATH)

    def create_first_stage(self, anchors = YOLO_ANCHORS):

        detectors_mask_shape = (RESOLUTION[0] // 16, RESOLUTION[1] // 16, 5, 1)
        matching_boxes_shape = (RESOLUTION[0] // 16, RESOLUTION[1] // 16, 5, 5)

        # Create model input layers.
        image_input = Input(batch_shape=(self.batch_size, self.timestep, RESOLUTION[0], RESOLUTION[1], RESOLUTION[2]))
        boxes_input = Input(shape=(None, 5))
        detectors_mask_input = Input(shape=detectors_mask_shape)
        matching_boxes_input = Input(shape=matching_boxes_shape)

        # Create model body.
        yolo_model = yolo_body(image_input, len(anchors), self.n_classes, self.batch_size, 0)


        # Place model loss on CPU to reduce GPU memory usage.
        with tf.device('/cpu:0'):
            # TODO: Replace Lambda with custom Keras layer for loss.
            model_loss = Lambda(
                yolo_loss,
                output_shape=(1,),
                name='yolo_loss',
                arguments={'anchors': anchors, 'num_classes': self.n_classes})([yolo_model.output, boxes_input, detectors_mask_input, matching_boxes_input])

        model = Model(
            [yolo_model.input, boxes_input, detectors_mask_input,
             matching_boxes_input], model_loss)

        return yolo_model, model

    def extract_data(self, filepath):

        n_test_files = 2
        lst_data = self.get_results(filepath, [])

        num_files = len(lst_data)
        num_imgs = [0] * num_files
        data_out = [0] * num_files

        for i, file in enumerate(lst_data):
            print("{}/{}, processing {}".format(i,num_files, file))
            df = pd.read_csv(file, index_col=0, header=None)
            count = 0
            lst_out = []
            for index, row in df.iterrows():
                fileparts = index.split('\\')[-2:]
                name = os.path.join(filepath, fileparts[0],fileparts[1])
                row_val = pd.Series.dropna(row).values
                lst_out.append([name, row_val])
                count += 1

            data_out[i] = lst_out
            num_imgs[i] = count



        training_generator = DataGenerator(data_out[:1 - n_test_files], num_files - n_test_files, num_imgs[:1 - n_test_files],  YOLO_ANCHORS, 0, self.batch_size, shuffle=True)
        validation_generator = DataGenerator(data_out[-1 * n_test_files:], n_test_files, num_imgs[-1 * n_test_files:], YOLO_ANCHORS, 0, self.batch_size, shuffle=False)

        return training_generator, validation_generator

    def __get_data(self, root_folder, lst_img, lst_box):

        os.chdir(root_folder)

        for elem in os.listdir():

            if os.path.isdir(elem):
                # print(elem)
                lst_img, lst_box = self.__get_data(elem, lst_img, lst_box)
                os.chdir('..')
            else:
                if ".npy" in elem:
                    if 'data_img' in elem:
                        lst_img.append(os.path.abspath(elem))
                    elif 'data_box' in elem:
                        lst_box.append(os.path.abspath(elem))

        return lst_img, lst_box

    def get_results(self, root_folder, lst):

        os.chdir(root_folder)

        for elem in os.listdir():

            if os.path.isdir(elem):
                # print(elem)
                lst = self.get_results(elem, lst)
                os.chdir('..')
            else:
                if "Final" in elem and ".csv" in elem:
                    lst.append(os.path.abspath(elem))

        return lst

    def train_model(self, stage):


        checkpoint = ModelCheckpoint(RESTORE_PATHS[0], monitor='val_loss', save_weights_only=True, save_best_only=False)
        checkpoint_best = ModelCheckpoint(RESTORE_PATHS[1], monitor='val_loss', save_weights_only=True, save_best_only=True)

        if stage == 0:
            if self.stateful:
                self.model_first.fit_generator(generator=self.training_generator, validation_data=self.validation_generator, epochs=1000, verbose=1, callbacks=[checkpoint, checkpoint_best, plot_losses], shuffle=False)
                self.model.save_weights(RESTORE_PATHS[0])
            else:
                self.model_first.fit_generator(generator=self.training_generator, validation_data=self.validation_generator, epochs=1000, verbose=1, callbacks=[checkpoint, checkpoint_best, plot_losses], shuffle=False)
                self.model.save_weights(RESTORE_PATHS[0])