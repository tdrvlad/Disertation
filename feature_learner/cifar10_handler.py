

import tensorflow as tf


LABEL_MAP = {
    0: 'airplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck'
}


def get_image_detections_and_detection_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    image_detections = {}
    detection_data = {}

    for i in range(x_train.shape[0]):
        bbox_dict = {
            'category_id': int(y_train[i])
        }
        
        image_detections[i] = {
            i: bbox_dict
        }
        
        detection_data[i] = x_train[i]

    return image_detections, detection_data
