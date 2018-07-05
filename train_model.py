"""
Anton Boykov
This is a script that can be used to train the gesture recognition model
Script based on code from: https://github.com/allanzelener/YAD2K
"""

from Gesture_Localizer import Gesture_Localizer
from data_processing import get_classes

#getting defaults
from parameters.default_values import IMAGESIZE, RESOLUTION, RESTORE_PATHS, DATAPATH, N_CLASSES, YOLO_ANCHORS

#main script from which the training is conducted
def _main():
    batch_size = 20
    timestep = 1
    lr = 0.0001
    dc = 0.01
    a = Gesture_Localizer(N_CLASSES[0], batch_size, timestep, lr=lr, dc=dc)
    print(a.model_first.summary())
    a.train_model(0)

    classes_path = "C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\topic\\machine_learning\\mouse_controller\\Final_Version\\parameters\\classes.txt"
    class_names = get_classes(classes_path)









if __name__ == '__main__':
    _main()