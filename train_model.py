"""
Anton Boykov
This is a script that can be used to train the gesture recognition model
Script based on code from: https://github.com/allanzelener/YAD2K
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import random

from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.optimizers import  Nadam
from keras.utils import Sequence



from data_augmentor import augment, check_labels, change_fov
from data_processing import DataGenerator, process_data, get_detector_mask
from yolo_body import (preprocess_true_boxes, yolo_body, yolo_eval, yolo_head, yolo_loss)
from draw_boxes import draw_boxes
from visualisation import PlotLearning

#getting defaults
from parameters.default_values import IMAGESIZE, RESOLUTION, RESTORE_PATHS, N_CLASSES, YOLO_ANCHORS




print(IMAGESIZE)


    # def __data_generation(self, list_IDs_temp):
    #     'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    #     # Initialization
    #     X = np.empty((self.batch_size, *self.dim, self.n_channels))
    #     y = np.empty((self.batch_size), dtype=int)
    #
    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #         # Store sample
    #         X[i,] = np.load('data/' + ID + '.npy')
    #
    #         # Store class
    #         y[i] = self.labels[ID]
    #
    #     return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

plot_losses = PlotLearning()

# plot_losses = PlotLosses()
# plot_losses = livelossplot.PlotLossesKeras()

# helper = lambda i, b, a: [i, b].append(get_detector_mask(b, a))
# process = lambda x, y, a: helper(process_data(x, y), a)


def _main():

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

def create_model(anchors, class_names, batch_size, timestep, weights, load_pretrained=False, freeze_body=False):
    '''
    returns the body of the model and the model
    # Params:
    load_pretrained: whether or not to load the pretrained model or initialize all weights
    freeze_body: whether or not to freeze all weights except for the last layer's
    # Returns:
    model_body: YOLOv2 with new output layer
    model: YOLOv2 with custom loss Lambda layer
    '''

    detectors_mask_shape = (RESOLUTION[0]//32, RESOLUTION[1]//32, 5, 1)
    matching_boxes_shape = (RESOLUTION[0]//32, RESOLUTION[1]//32, 5, 5)

    # Create model input layers.
    image_input = Input(batch_shape=(batch_size, RESOLUTION[0], RESOLUTION[1], RESOLUTION[2]))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    # topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    # yolo_model.load_weights(weights)
    #
    # reg = regularizers.l1(3)
    # constr = min_max_norm(min_value=-0.5, max_value=0.5)
    #
    # x = yolo_model.output
    # x = Reshape((1, 7, 7, 45))(x)
    # x = ConvLSTM2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear', kernel_initializer= 'glorot_uniform', kernel_regularizer=reg , recurrent_regularizer=reg , bias_regularizer=reg , activity_regularizer=reg , kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, stateful=False)(x)
    # x = Reshape((7, 7, 45))(x)

    if load_pretrained:
        yolo_model = load_model(load_pretrained)

    if freeze_body:
        for layer in yolo_model.layers[0:-2]:
            layer.trainable = False

    # final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)


    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           yolo_model.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [yolo_model.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return yolo_model, model


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

    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

#lol
    # for i in range(100):
    #     model.fit_generator(generator=generator, #,([image_data, boxes, detectors_mask, matching_true_boxes],
    #               #np.zeros(len(image_data)),
    #               #validation_split=validation_split,
    #               # batch_size=batch_size,
    #               steps_per_epoch=100,
    #               epochs=5,
    #               shuffle=False,
    #               # use_multiprocessing = True,
    #               callbacks=[logging])
    #     model.save_weights(RESTORE)
    #
    #     preds = model.evaluate_generator(
    #               # [image_data, boxes, detectors_mask, matching_true_boxes],
    #               # np.zeros(len(image_data)),
    #               generator=generator,
    #               # batch_size=batch_size,
    #               verbose=1,
    #               sample_weight=None)
    #
    #     print()
    #     print("Loss = " + str(preds[0]))
    #     print("Test Accuracy = " + str(preds[1]))



def draw(model_body, class_names, anchors, image_data, image_set='train',
            weights_name='model_save.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[:int(len(image_data)*.9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[int(len(image_data)*.9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
    # model_body.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()



if __name__ == '__main__':
    _main()