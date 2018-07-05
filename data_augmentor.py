import cv2
import numpy as np
import random
from draw_boxes import draw_boxes

from parameters.default_values import IMAGESIZE, RESOLUTION, RESTORE_PATHS, N_CLASSES, YOLO_ANCHORS

def augment(images, boxes, code, flip = -1, transpose = -1, crop = 0):
    # check_labels(images, boxes, ["lol", "lol", "lol", "lol", "lol", "lol", "lol", "lol", "lol", "lol"])
   # cv2.imshow('image', draw_boxes(images[0], [boxes[0].values[:4] * 224], [boxes[0].values[4].astype(int)], ["lol", "lol"]))

    # seed = np.random.seed(code)

    out_images = []
    out_boxes = []

    img_shape = images[0].shape
    crop_dims = [0] * 4
    if crop_dims:
        crop_dims[0] = img_shape[0] * crop[0]
        crop_dims[1] = img_shape[0]* crop[1]
        crop_dims[2] = img_shape[1]* crop[2]
        crop_dims[3] = img_shape[1]* crop[3]
        crop_dims[0] = min(max(0, round(crop_dims[0])), img_shape[0])
        crop_dims[1] = min(max(0, round(crop_dims[1])), img_shape[0])
        crop_dims[2] = min(max(0, round(crop_dims[2])), img_shape[1])
        crop_dims[3] = min(max(0, round(crop_dims[3])), img_shape[1])

    for image, box in zip(images, boxes):

        img_shape = image.shape

        # factor = (code + 1) * random.randint(1, 100) % 3 / 10
        if flip == -1:
            flip = (code + 1) * random.randint(1, 100) % 4
        if transpose == -1:
            transpose = (code + 1) * random.randint(1, 100) % 2

        # for i in range(2):
        #     image = noisy(noises[i], image, factor)
        if crop_dims:
            # cv2.imshow('image', image)
            # cv2.waitKey(0)
            image = image[crop_dims[0]: crop_dims[1], crop_dims[2]:crop_dims[3], :]
            # print(image.shape)
            # print(image.shape, crop_dims)
            # cv2.imshow('a', image[:, :, 0].astype(np.uint8))
            # cv2.waitKey(0)
            image = cv2.resize(image, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_AREA)
            # print(image.shape)
            # cv2.imshow('image', image)
            # cv2.waitKey(0)

        if flip != 0:
            image = cv2.flip(image, flip - 2)
        if transpose != 0:
            center = (img_shape[1] // 2, img_shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, 90, 1)
            image = cv2.warpAffine(image, M, (img_shape[0], img_shape[1]))

        out_images.append(image)
        out_boxes.append(box_adjust(box, flip, transpose, [crop_dims, img_shape]))

    # check_labels(out_images, out_boxes, ["lol", "lol", "lol", "lol", "lol", "lol", "lol", "lol", "lol", "lol"], 1)

    return [out_images, out_boxes]


def crop_process(boxes, dims):
    out = []
    num_boxes = 0
    [crop_dims, image_dims] = dims

    for box in boxes:
        x1 = box[0] * image_dims[1]
        y1 = box[1] * image_dims[0]
        x2 = box[2] * image_dims[1]
        y2 = box[3] * image_dims[0]

        area1 = (box[2] - box[0]) * (box[3] - box[1])

        x1 = min(max(x1, crop_dims[2]), crop_dims[3])
        x2 = min(max(x2, crop_dims[2]), crop_dims[3])
        y1 = min(max(y1, crop_dims[0]), crop_dims[1])
        y2 = min(max(y2, crop_dims[0]), crop_dims[1])

        height = crop_dims[1] - crop_dims[0]
        width = crop_dims[3] - crop_dims[2]

        if x1 != x2 and y1 != y2:

            x1 = (x1 - crop_dims[2]) / width
            y1 = (y1 - crop_dims[0]) / height
            x2 = (x2 - crop_dims[2]) / width
            y2 = (y2 - crop_dims[0]) / height

            area2 = (x2 - x1) * (y2 - y1)

            if area2/area1 > 0.6:

                num_boxes += 1

                out.append([x1, y1, x2, y2, box[4]])

    result = np.zeros((num_boxes, 5))

    for i, box in enumerate(out):
        for j in range(5):
            result[i][j] = box[j]

    return result




def box_adjust(boxes, flip, transpose, dims):


    if len(boxes):
        boxes = boxes.reshape((-1, 5))
        if dims[0]:
            boxes = crop_process(boxes, dims)
        out_boxes = np.zeros((boxes.shape[0], boxes.shape[1]))
    else:
        out_boxes = boxes

    count = 0
    for box in boxes:

        if flip == 1 or flip == 3:
            box[0] = 1 - box[0]
            box[2] = 1 - box[2]

        if flip == 1 or flip == 2:
            box[1] = 1 - box[1]
            box[3] = 1 - box[3]

        if transpose:
            x1 = box[1]
            y1 = 1 - box[0]
            x2 = box[3]
            y2 = 1 - box[2]
            box = np.array([x1,y1,x2,y2, box[4]])

        out_boxes[count] = (reformat_box(box))
        count += 1

    return out_boxes

def reformat_box(box):

    new_box = np.zeros(5,)

    new_box[0] = min(box[0], box[2])
    new_box[1] = min(box[1], box[3])
    new_box[2] = max(box[0], box[2])
    new_box[3] = max(box[1], box[3])
    new_box[4] = box[4]

    return new_box


noises = ['gauss',  's&p']#['gauss', 'poisson', 's&p', 'speckle']

def noisy(noise_typ,image, factor):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        # var = 0.2
        # sigma = var**0.5
        sigma = 1.0
        gauss = np.random.normal(mean,sigma,(row,col,ch)) * factor
        gauss = gauss.reshape(row,col,ch)
        noisy = cv2.addWeighted(gauss.astype(np.uint8),1.0,image.astype(np.uint8), 1.0, 0.0)

        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy.astype(np.uint8)
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss * factor
        return noisy.astype(np.uint8)

def check_labels(images, boxes, class_names, augmented = 0):
    for img, box in zip(images, boxes):

        if len(box):
            if not augmented:
                box = box.values
            box = box.reshape((-1, 5))
            box = change_fov(box)
            img_dims = img.shape
            cv2.imshow('image', draw_boxes(img[:,:,0] /255, box[:, :4] * RESOLUTION[0], box[:, 4].astype(int), class_names))
            cv2.imshow('image', draw_boxes(img[:,:,1] /255, box[:, :4] * RESOLUTION[0], box[:, 4].astype(int), class_names))
            cv2.waitKey(60)
        else:
            cv2.imshow('image', img[:,:,0])
            cv2.imshow('image', img[:,:,1])
            cv2.waitKey(20)

def change_fov(boxes, wh = 0):

    out = np.zeros((boxes.shape[0], boxes.shape[1]))
    count = 0
    for box in boxes:
        if wh:
            y1 = box[0] - box[2] / 2
            x1 = box[1] - box[3] / 2
            y2 = box[0] + box[2] / 2
            x2 = box[1] + box[3] / 2
        else:
            x1 = box[1]
            y1 = box[0]
            x2 = box[3]
            y2 = box[2]
        out[count] = np.array([x1,y1, x2, y2, box[4]])
        count += 1
    return out