import numpy as np

IMAGESIZE = 150528
RESOLUTION = [224, 224, 3]
RESTORE_PATHS = ['Weights\\model_progres_seg1.h5', 'Weights\\model_progres_seg2.h5', 'Weights\\model_progres_seg3.h5']
N_CLASSES = 3
YOLO_ANCHORS = np.array(((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),(7.88282, 3.52778), (9.77052, 9.16828)))
