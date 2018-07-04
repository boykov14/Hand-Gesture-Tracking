"""
Anton Boykov
This is a script that can be used to train the gesture recognition model
Script based on code from: https://github.com/allanzelener/YAD2K
"""

import os
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Input, Lambda, ConvLSTM2D, Reshape
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.optimizers import  Nadam
from keras.regularizers import l2

#getting necessary values from local files
from data_processing import DataGenerator, process_data, get_detector_mask
from yolo_body import (preprocess_true_boxes, yolo_body, yolo_eval, yolo_head, yolo_loss)
from visualisation import PlotLearning, draw_boxes

#getting defaults
from parameters.default_values import IMAGESIZE, RESOLUTION, RESTORE_PATHS, N_CLASSES, YOLO_ANCHORS

#setting up plotting class
plot_losses = PlotLearning()

#defining class which we will use for creating/manipulating model
class Gesture_Localizer():

    def __init__(self, n_classes, batch_size, timestep, lr = 0.0001, dc = 0.004):
        self.RESOLUTION = RESOLUTION
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.timestep = timestep

        optimizer = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=dc)
        [yolo_model_first, self.model_first] = self.create_first_stage()
        self.model_first.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}, metrics=['accuracy'])

    def create_first_stage(self, anchors = YOLO_ANCHORS):

        detectors_mask_shape = (RESOLUTION[0] // 8, RESOLUTION[1] // 8, 5, 1)
        matching_boxes_shape = (RESOLUTION[0] // 8, RESOLUTION[1] // 8, 5, 5)

        # Create model input layers.
        image_input = Input(batch_shape=(self.batch_size, self.timestep, RESOLUTION[0], RESOLUTION[1], RESOLUTION[2]))
        boxes_input = Input(shape=(None, 5))
        detectors_mask_input = Input(shape=detectors_mask_shape)
        matching_boxes_input = Input(shape=matching_boxes_shape)

        # Create model body.
        yolo_model = yolo_body(image_input, len(anchors), self.n_classes, self.batch_size, 0)

        reg = l2(5e-4)

        x = yolo_model.output
        dims = x.shape
        x = Reshape((1, int(dims[1]), int(dims[2]), int(dims[3])))(x)
        x = ConvLSTM2D(len(anchors)*(5+self.n_classes), (1, 1), activation='linear', kernel_initializer= 'glorot_uniform', kernel_regularizer=reg , recurrent_regularizer=reg , bias_regularizer=reg , activity_regularizer=reg, stateful=False)(x)
        x = Reshape((int(dims[1]), int(dims[2]), int(dims[3])))(x)


        # Place model loss on CPU to reduce GPU memory usage.
        with tf.device('/cpu:0'):
            # TODO: Replace Lambda with custom Keras layer for loss.
            model_loss = Lambda(
                yolo_loss,
                output_shape=(1,),
                name='yolo_loss',
                arguments={'anchors': anchors, 'num_classes': self.n_classes})([x, boxes_input, detectors_mask_input, matching_boxes_input])

        model = Model(
            [yolo_model.input, boxes_input, detectors_mask_input,
             matching_boxes_input], model_loss)

        return yolo_model, model

    def extract_data(self, filepath):

#main script from which the training is conducted
def _main():
    a = Gesture_Localizer(N_CLASSES, 10, 1)
    print(a.model_first.summary())

    data_path = "D:\\machine_learning\\mouse_control\\data\\combined_data.npy"
    # best_model_path = 'model_best_new.h5'
    best_model_path = 'model_best_new_class.h5'
    model_save = 'test.h5'
    model_path = '20180626_model_path.h5'

    classes_path = "parameters\\classes.txt"
    class_names = get_classes(classes_path)

    batch_size = 4
    timestep = 1
    dataset_iters = 0

    #[images, boxes] = np.load(data_path) # custom data saved as a numpy file.
    #  has 2 arrays: an object array 'boxes' (variable length of boxes in each image)
    #  and an array of images 'images'

    images_training = []
    boxes_training = []
    names_img = []
    names_box = []
    for i in range(0,15):
        print("{}\{} ".format(i+1, 15))
        names_img.append(os.path.join('C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\mouse_controller\\Sequences', ('data_img' + str(i + 1) + '.npy')))
        names_box.append(os.path.join('C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\mouse_controller\\Sequences', ('data_box' + str(i + 1) + '.npy')))

        images_training.append(np.load(os.path.join('C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\mouse_controller\\Sequences', ('data_img' + str(i + 1) + '.npy')))) #np.concatenate([images_training, np.load(os.path.join('C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\mouse_controller\\Sequences', ('data_img' + str(i + 1) + '.npy')))])
        boxes_training.append(np.load(os.path.join('C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\mouse_controller\\Sequences', ('data_box' + str(i + 1) + '.npy'))))#np.concatenate([boxes_training, np.load(os.path.join('C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\mouse_controller\\Sequences', ('data_box' + str(i + 1) + '.npy')))])
        #check_labels(images_training[i], boxes_training[i], class_names)


    anchors = YOLO_ANCHORS


    images_val = np.load('C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\mouse_controller\\Sequences\\data_img_val.npy')
    boxes_val =  np.load('C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\mouse_controller\\Sequences\\data_box_val.npy')

    training_generator = DataGenerator(images_training, boxes_training, anchors, batch_size, shuffle = False)
    validation_generator = DataGenerator([images_val], [boxes_val], anchors, batch_size, shuffle = False)

    model_body, model = create_model(anchors, class_names, batch_size, timestep, best_model_path)

    optimizer = Nadam(lr=0.000009, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.01)
    model.compile(
        optimizer=optimizer , loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        }, metrics=['accuracy'])  # This is a hack to use the custom loss function in the last layer.

    # model.layers[-1]().load_weights(best_model_path)
    #model.load_weights(best_model_path)
    train_gen(model, training_generator, validation_generator, best_model_path)



    image_data, boxes = process_data(images_val, boxes_val)

    # detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)

    model_body, model = create_model(anchors, class_names, 1, timestep, weights=best_model_path)

    model.compile(
        optimizer=optimizer, loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        }, metrics=['accuracy'])  # This is a hack to use the custom loss function in the last layer.

    model.load_weights(best_model_path)

    for i in range(15):
        im, _ = process_data(images_training[i], boxes_training[i])
        draw(model_body,
            class_names,
            anchors,
            im,#image_data,
            out_path="output_images" + str(i),
            image_set='all', # assumes training/validation split is 0.9
            weights_name=best_model_path,
            save_all=False)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS




def train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, savepath, batch_size, validation_split=0.1):
    # optimizer = Nadam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    # model.compile(
    #     optimizer=optimizer, loss={
    #         'yolo_loss': lambda y_true, y_pred: y_pred
    #     }, metrics=['accuracy'])  # This is a hack to use the custom loss function in the last layer.
    #
    # model.load_weights(RESTORE)

    checkpoint = ModelCheckpoint(savepath, monitor='val_loss', save_weights_only=True, save_best_only=True)
    tb = K.callbacks.TensorBoard(log_dir='./tb_logs', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True)

    logging = checkpoint

    for i in range(1):
        history = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                   np.zeros(len(image_data)),
                   validation_split=validation_split,
                   batch_size=batch_size,
                   epochs=10,
                   shuffle=True,
                    verbose=1,
                   callbacks=[logging, tb, plot_losses])

        model.save_weights(RESTORE)

        # preds = model.evaluate(
        #     [image_data, boxes, detectors_mask, matching_true_boxes],
        #     np.zeros(len(image_data)),
        #     batch_size=batch_size,
        #     verbose=1,
        #     sample_weight=None)
        #
        # print()
        # print("Loss = " + str(preds[0]))
        # print("Test Accuracy = " + str(preds[1]))


def train_gen(model, training_generator, validation_generator, savepath):
    '''
    retrain/fine-tune the model
    logs training with tensorboard
    saves training weights in current directory
    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''

    # model.load_weights(savepath)

     #TensorBoard()
    # checkpoint = ModelCheckpoint(savepath, monitor='val_loss',
    #                              save_weights_only=True, save_best_only=True)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    checkpoint = ModelCheckpoint(savepath, monitor='val_loss', save_weights_only=True)
    # tb = K.callbacks.TensorBoard(log_dir='./tb_logs', histogram_freq=1, batch_size=20, write_graph=True,
    #                                  write_grads=True, write_images=True)

    logging = checkpoint

    # model.save_weights(savepath)
    model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=1000, verbose=1, callbacks = [logging, plot_losses], shuffle=True)
    model.save_weights(savepath)



if __name__ == '__main__':
    _main()