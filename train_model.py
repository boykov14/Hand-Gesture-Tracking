"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import keras as K
import random
from tensorflow import floor, mod
from keras.utils import to_categorical
from keras import regularizers
from keras.constraints import min_max_norm
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D, Reshape
from keras.layers import ConvLSTM2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
from keras.optimizers import adam, Nadam, Adamax
from IPython.display import clear_output
from keras.optimizers import Optimizer
from keras.utils import Sequence
from keras.utils import Sequence
from keras.initializers import glorot_uniform
from skimage.io import imread
from skimage.transform import resize
import livelossplot

from data_augmentor import augment, check_labels, change_fov

import time

from YOLO import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from draw_boxes import draw_boxes

IMAGESIZE = 150528
RESOLUTION = [224, 224, 3]
RESTORE = 'model_progres.h5'
N_CLASSES = 3

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))


class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        plt.ion()

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        clear_output(wait=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()

        plt.show();
        plt.pause(10)
        plt.close('all')
        print('continue computation')
        # at the end call show to ensure window won't close.
        #plt.show()

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, names_imgs, names_boxes, anchors, batch_size=10, shuffle = False, timestep=1):
        'Initialization'
        self.shuffle = shuffle
        self.names_imgs = names_imgs
        self.names_boxes = names_boxes
        self.batch_size = batch_size
        self.dataset_sizes = []
        self.anchors = anchors
        self.timestep = timestep
        self.epoch_index = 0
        self.flip = random.randint(1, 100) % 4
        self.transpose = random.randint(1, 100) % 2

        self.num_imgs = 0
        for i in range(len(names_imgs)):
            size = len(names_imgs[i]) // batch_size * batch_size

            self.dataset_sizes.append(size)
            self.num_imgs += size
            self.names_imgs[i] = self.names_imgs[i][:size]
            self.names_boxes[i] = self.names_boxes[i][:size]

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'

        return int(np.floor(self.num_imgs / self.batch_size))

    def __getitem__(self, index):
        # if self.epoch_index ==0:
        #     print("epoch ___1___: index {}".format(index))
        # else:
        #print("epoch index {}".format(index))
        'Generate one batch of data'
        # Generate indexes of the batch
        # print(index)
        # print("    {}".format(index))
        index = self.indexes[index]
        # print("    {}".format(index))
        dataset_index = 0
        internal_index = 0
        for i in range(0, len(self.dataset_sizes)):
            dataset_index += self.dataset_sizes[i]
            if dataset_index > index * self.batch_size:
                internal_index = index * self.batch_size - dataset_index + self.dataset_sizes[i]
                if internal_index == 0:
                    self.flip = random.randint(1, 100) % 4
                    self.transpose = random.randint(1, 100) % 2
                dataset_index = i
                break

        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #
        # # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_images = self.names_imgs[dataset_index][internal_index:internal_index + self.batch_size]
        list_boxes = self.names_boxes[dataset_index][internal_index:internal_index + self.batch_size]

        self.flip = random.randint(1, 100) % 4
        self.transpose = random.randint(1, 100) % 2

        list_images, list_boxes = augment(list_images , list_boxes, (index * self.epoch_index), self.flip, self.transpose)

        image_data, boxes = process_data(list_images , list_boxes, 1)
        detectors_mask, matching_true_boxes = get_detector_mask(boxes, self.anchors)

        # Generate data
        X, y = [image_data, boxes, detectors_mask, matching_true_boxes], np.zeros(len(image_data)) #((-1, self.timestep, RESOLUTION[0], RESOLUTION[1], RESOLUTION[2]))), boxes, detectors_mask, matching_true_boxes], np.zeros(len(image_data))

        return X, y

    def on_epoch_end(self):
        self.flip = random.randint(1, 100) % 4
        self.transpose = random.randint(1, 100) % 2

        'Updates indexes after each epoch'
        self.indexes = np.arange(int(np.floor(self.num_imgs / self.batch_size)))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.epoch_index += 1

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

    classes_path = "classes.txt"
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


    images_val = np.load('data_img.npy')
    boxes_val = np.load('data_box.npy')

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

def process_data(images, boxes=None, augmented = 0):
    '''processes the data'''
    images = [PIL.Image.fromarray(i,'RGB') for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [i.resize((RESOLUTION[0], RESOLUTION[1]), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        if not augmented:
            boxes = [(box.values).reshape((-1, 5)) for box in boxes]
        else:
            boxes = [box.reshape((-1, 5)) for box in boxes]

        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [1, 0, 3, 2, 4]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 2:4] + box[:, 0:2]) for box in boxes]
        boxes_wh = [box[:, 2:4] - box[:, 0:2] for box in boxes]
        boxes_xy = [boxxy for boxxy in boxes_xy]
        boxes_wh = [boxwh for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 4:5]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [RESOLUTION[0], RESOLUTION[1]])

    return np.array(detectors_mask), np.array(matching_true_boxes)

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