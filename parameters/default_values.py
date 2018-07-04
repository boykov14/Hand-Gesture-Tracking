import numpy as np

IMAGESIZE = 200704
RESOLUTION = [112, 112, 4]
RESTORE_PATHS = ['Weights\\model_progres_seg1.h5', 'Weights\\model_best_seg1.h5',
                 'Weights\\model_progres_seg2.h5', 'Weights\\model_best_seg2.h5',
                 'Weights\\model_progres_seg3.h5', 'Weights\\model_best_seg3.h5']
DATAPATH = 'C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\mouse_controller\\Sequences'
N_CLASSES = [1,3]
YOLO_ANCHORS = np.array(((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),(7.88282, 3.52778), (9.77052, 9.16828)))
