import numpy as np
import random
import PIL

from keras.utils import Sequence
from data_augmentor import augment, check_labels
from yolo_body import preprocess_true_boxes

# Setting Default Values
from parameters.default_values import IMAGESIZE, RESOLUTION, RESTORE_PATHS, N_CLASSES, YOLO_ANCHORS

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, names_imgs, names_boxes, anchors, stage, batch_size=10, shuffle = False, timestep=1):
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
        self.stage = stage

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

        image_data, boxes = process_data(list_images, self.stage, list_boxes, 1)
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

def process_data(images, stage, boxes=None, augmented=0):
    '''processes the data'''
    images = [PIL.Image.fromarray(i, 'RGB') for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [i.resize((RESOLUTION[0], RESOLUTION[1]), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image / 255. for image in processed_images]
    processed_images = np.reshape(processed_images, (-1, 1, RESOLUTION[0], RESOLUTION[1], RESOLUTION[2]))

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