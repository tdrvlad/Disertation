from model_handler import *
from data_handler import *
import tensorflow as tf

create_doublehead_model(input_shape = (32,32,3), embedding_size = 128, hidden_layer_neurons = 0, model_name = 'doublehead_model')


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

cifar_image_detections = {}
cifar_detection_data = {}

for i in range(x_train.shape[0]):
    bbox_dict = {
        'category_id': int(y_train[i])
    }
    
    cifar_image_detections[i] = {
        i: bbox_dict
    }
    
    cifar_detection_data[i] = x_train[i]

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

cifar_detection_data = cifar_detection_data.copy() 

for k,v in cifar_detection_data.items():
    cifar_detection_data[k] = v / 255.

target_classes = None

data_handler_obj = DataHandler(
    cifar_image_detections, 
    cifar_detection_data, 
    target_classes = target_classes, 
    label_map = LABEL_MAP,
    apply_preprocessing = True
)

data_handler_obj.add_augmentation(flip=True, rotation=True, translation=True, zoom=True, contrast=False)

all_object_categories = data_handler_obj.get_all_object_categories()
print(len(all_object_categories), set(all_object_categories))


train_model(
    'doublehead_model', 
    data_handler_obj, 
    new_model_name = 'autoencoder', 
    no_epochs = 200, 
    steps_per_epoch = None, 
    learning_rate = 0.0001, 
    entity_loss_weight = 0, 
    context_loss_weight = 0,
    reconstruction_loss_weight = 100,
    freeze_backbone = False, 
    doublehead = True
)