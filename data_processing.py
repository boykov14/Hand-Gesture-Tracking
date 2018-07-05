import pandas as pd
import numpy as np
import random
import PIL

from keras.utils import Sequence
from data_augmentor import augment, check_labels
from yolo_body import preprocess_true_boxes
import cv2

# Setting Default Values
from parameters.default_values import IMAGESIZE, RESOLUTION, RESTORE_PATHS, N_CLASSES, YOLO_ANCHORS

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, img_files, num_files, count_imgs, anchors, stage, batch_size=10, shuffle = False, timestep=1):
        'Initialization'
        self.shuffle = shuffle
        self.img_files = img_files
        self.num_files = num_files
        self.count_imgs = count_imgs
        self.batch_size = batch_size
        self.dataset_sizes = []
        self.anchors = anchors
        self.timestep = timestep
        self.epoch_index = 0
        self.stage = stage
        self.data_im = [0] * 20
        self.data_bo = [0] * 20
        self.prev = np.zeros((RESOLUTION[0], RESOLUTION[1], 1)).astype(np.uint8)
        self.diff = np.zeros((RESOLUTION[0], RESOLUTION[1], 1)).astype(np.uint8)

        self.num_imgs = 0
        for i in range(num_files):
            size = len(img_files[i]) // batch_size * batch_size

            self.dataset_sizes.append(size)
            self.num_imgs += size

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'

        return int(np.floor(self.num_imgs / self.batch_size))

    def __getitem__(self, index):

        #index = self.indexes[index]
        # print("    {}".format(index))
        dataset_index = 0
        internal_index = 0
        for i in range(0, len(self.dataset_sizes)):
            ii = self.dataset_indexes[i]
            dataset_index += self.dataset_sizes[ii]
            if dataset_index > index * self.batch_size:
                internal_index = index * self.batch_size - dataset_index + self.dataset_sizes[ii]
                dataset_index = ii
                break

        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #
        # # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_data = self.img_files[dataset_index][internal_index:internal_index + self.batch_size]

        if internal_index == 0:
            self.prev = np.zeros((RESOLUTION[0], RESOLUTION[1], 1)).astype(np.uint8)
            self.diff = np.zeros((RESOLUTION[0], RESOLUTION[1], 1)).astype(np.uint8)

        list_images = []
        list_boxes = []

        for val in list_data:
            img = cv2.imread(val[0])
            img = self.reformat(img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            diff = cv2.subtract(gray, self.prev)
            A = np.zeros((RESOLUTION[0], RESOLUTION[1], 1))
            A[diff > 10] = 1
            diff = A
            self.prev = gray

            result = np.concatenate((np.expand_dims(gray, axis=2), diff), axis=2)
            list_images.append(result)
            list_boxes.append(val[1])


        # list_images = self.data_im[dataset_index][internal_index:internal_index + self.batch_size]
        # list_boxes = self.data_bo[dataset_index][internal_index:internal_index + self.batch_size]


        flip = self.flip[dataset_index]
        transpose = self.transpose[dataset_index]
        crop = self.crop[dataset_index]

        list_images, list_boxes = augment(list_images , list_boxes, (index * self.epoch_index), flip, transpose, crop)

        for im in list_images:
            cv2.imshow('a', im[:,:,0].astype(np.uint8))
            cv2.imshow('b', im[:,:,1])
            cv2.waitKey(60)

        image_data, boxes = process_data(list_images, self.stage, list_boxes, 1)
        detectors_mask, matching_true_boxes = get_detector_mask(boxes, self.anchors)


        # Generate data
        X, y = [image_data, boxes, detectors_mask, matching_true_boxes], np.zeros(len(image_data)) #((-1, self.timestep, RESOLUTION[0], RESOLUTION[1], RESOLUTION[2]))), boxes, detectors_mask, matching_true_boxes], np.zeros(len(image_data))

        return X, y

    def reformat(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (RESOLUTION[0], RESOLUTION[1]), interpolation=cv2.INTER_AREA)
        return frame

    def on_epoch_end(self):

        self.flip = [0] * self.num_files
        self.transpose = [0] * self.num_files
        self.crop = [0] * self.num_files

        for i in range(self.num_files):

            self.flip[i] = random.randint(1, 100) % 4
            self.transpose[i] = random.randint(1, 100) % 2

            x1 = random.randint(0, round(RESOLUTION[0] / 4)) / RESOLUTION[0]
            w1 = random.randint(round(RESOLUTION[0] / 2), RESOLUTION[0]) / RESOLUTION[0]
            y1 = random.randint(0, round(RESOLUTION[1] / 4)) / RESOLUTION[1]
            h1 = random.randint(round(RESOLUTION[1] / 2), RESOLUTION[1]) / RESOLUTION[1]

            crop = [y1, y1 + h1, x1, x1 + w1]
            self.crop[i] = crop

        'Updates indexes after each epoch'
        self.indexes = np.arange(int(np.floor(self.num_imgs / self.batch_size)))
        self.dataset_indexes = np.arange(self.num_files)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.dataset_indexes)
        self.epoch_index += 1

def process_data(images, stage, boxes=None, augmented=0):
    '''processes the data'''
    # images = [PIL.Image.fromarray(i, 'RGBA') for i in images]
    # orig_size = np.array([images[0].width, images[0].height])
    # orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    # processed_images = [i.resize((RESOLUTION[0], RESOLUTION[1]), PIL.Image.BICUBIC) for i in images]
    # processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    # processed_images = [image / 255. for image in processed_images]
    processed_images = np.reshape(images, (-1, 1, RESOLUTION[0], RESOLUTION[1], RESOLUTION[2]))

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        if not augmented:
            boxes = [(box.values).reshape((-1, 5)) for box in boxes]
        else:
            boxes = [box.reshape((-1, 5)) for box in boxes]

        if stage == 0:
            for i in range(len(boxes)):
                for j in range(len(boxes[i])):
                    boxes[i][j][4] = 0


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
            if boxz.shape[0] < max_boxes:
                zero_padding = np.zeros((max_boxes - boxz.shape[0], 5), dtype=np.float32)
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