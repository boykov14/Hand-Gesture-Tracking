import cv2
import numpy as np
import random
from draw_boxes import draw_boxes

IMAGESIZE = 150528
RESOLUTION = [224, 224, 3]

def augment(images, boxes, code, flip = -1, transpose = -1):
    # check_labels(images, boxes, ["lol", "lol", "lol", "lol", "lol", "lol", "lol", "lol", "lol", "lol"])


    # seed = np.random.seed(code)

    out_images = []
    out_boxes = []
    for image, box in zip(images, boxes):
        # factor = (code + 1) * random.randint(1, 100) % 3 / 10
        if flip == -1:
            flip = (code + 1) * random.randint(1, 100) % 4
        if transpose == -1:
            transpose = (code + 1) * random.randint(1, 100) % 2

        # for i in range(2):
        #     image = noisy(noises[i], image, factor)
        if flip:
            image = cv2.flip(image, flip - 2)
        if transpose:
            center = (RESOLUTION[0] // 2, RESOLUTION[1] // 2)
            M = cv2.getRotationMatrix2D(center, 90, 1)
            image = cv2.warpAffine(image, M, (RESOLUTION[0], RESOLUTION[1]))
        out_images.append(image)
        out_boxes.append(box_adjust(box, flip, transpose))
        # cv2.imshow('a', image[:,:,3])
        # cv2.waitKey(60)

    # check_labels(out_images, out_boxes, ["lol", "lol", "lol", "lol", "lol", "lol", "lol", "lol", "lol", "lol"], 1)

    return [out_images, out_boxes]

def box_adjust(boxes, flip, transpose):


    boxes = boxes.values


    if len(boxes):
        boxes = boxes.reshape((-1, 5))
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
            cv2.imshow('image', draw_boxes(img, box[:, :4] * 224, box[:, 4].astype(int), class_names))
        else:
            cv2.imshow('image', img)

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
