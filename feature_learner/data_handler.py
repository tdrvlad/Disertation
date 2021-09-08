
import numpy as np
import random, os, glob, io, json, math, yaml, time, sys
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw
from itertools import combinations 
from datetime import datetime
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import math
import re
import copy
import tensorflow as tf

class DataHandler:

    def __init__(self, image_detections, detection_data, target_classes = None, label_map = {}):

        self.label_map = label_map
        self.reverse_label_map = {v:k for k,v in self.label_map.items()}

        self.image_detections = self.filter_out_image_detections(image_detections, target_classes)
        self.detection_data = detection_data.copy()

        self.initialize_image_queue()
        self.augmentation_model = None


    def get_all_object_categories(self):
        
        categories = []
        for image_id, detections in self.image_detections.items():
            for detection_id, bbox in detections.items():
                categories.append(bbox['category_id'])
        return [self.label_map.get(c) for c in categories]


    def filter_out_image_detections(self, image_detections, target_classes = None):
        
        
        if target_classes is None:
            return image_detections

        target_classes = [self.reverse_label_map.get(c) for c in target_classes]
        
        image_detections_copy = {}
        for image_id, detections in image_detections.items():
            image_detections_copy[image_id] = detections.copy()
        
        image_detections = image_detections_copy

        image_ids_to_remove = []
        for image_id, detections in image_detections.items():
            detection_ids_to_remove = []

            for detection_id, bbox in detections.items():
                if bbox['category_id'] not in target_classes:
                    detection_ids_to_remove.append(detection_id)

            for detection_id in detection_ids_to_remove:
                del detections[detection_id]
        
            if len(image_detections[image_id]) == 0:
                image_ids_to_remove.append(image_id)
        
        for image_id in image_ids_to_remove:
            del image_detections[image_id]

        return image_detections


    def get_no_items(self):

        return len(self.detection_data)


    def initialize_image_queue(self):
        self.image_queue = list(self.image_detections.keys())
        random.shuffle(self.image_queue)


    def get_image_detections(self):

        if len(self.image_queue) == 0:
            self.initialize_image_queue()
        
        image_id = self.image_queue.pop()
        detections = self.image_detections.get(image_id)
        
        assert detections is not None, 'None detections.'

        return detections
        


    def create_batch(self, batch_size = 64, autoencoder_label = False, apply_imagenet_preprocessing = False, apply_normalization = False):

        batch_x = []
        batch_y = np.zeros((batch_size, 2))
       
        'label y has 2 dimensions: first is the entity id, second is context id'

        image_index = 0
        current_batch_size = 0
        
        while current_batch_size < batch_size:

            detections = self.get_image_detections()
        
            for detection_id, bbox in detections.items():
                if current_batch_size < batch_size:
                    
                    batch_x.append(self.detection_data.get(detection_id))
                    batch_y[current_batch_size][0] = bbox.get('category_id')
                    batch_y[current_batch_size][1] = image_index
                   
                    current_batch_size += 1

            image_index +=1
        
        batch_x = np.stack(batch_x)
       

        assert batch_x.shape[0] == batch_y.shape[0], 'x and y batches dont have the same size.'

        if not self.augmentation_model is None:
            batch_x = self.augmentation_model(batch_x, training = True)
        
        if apply_imagenet_preprocessing:
            batch_x = tf.keras.applications.mobilenet.preprocess_input(batch_x)
        
        if apply_imagenet_preprocessing:
            batch_x /= 255.

        if autoencoder_label:
            return batch_x, (batch_y, batch_x)
        else:
            return batch_x, batch_y

    
    def batch_generator(self, batch_size = 64, autoencoder_label = False, apply_imagenet_preprocessing = False, apply_normalization = False):

        while True:
            yield self.create_batch(batch_size, autoencoder_label = autoencoder_label, apply_imagenet_preprocessing, apply_normalization)


    def add_augmentation(self, flip=True, rotation=True, translation=True, zoom=True, contrast=False):
        '''
        Define data augmentation operations to be executed on data before being served.
        '''
        
        augmentation_model = tf.keras.Sequential(name = 'AugmentationModel')
        if flip:
            augmentation_model.add(tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"))
        if rotation:
            augmentation_model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(factor = 0.05, fill_mode = 'constant'))
        if translation:
            augmentation_model.add(tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode = 'constant'))
        if zoom:
            augmentation_model.add(tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor = 0.2, fill_mode = 'constant'))
        if contrast:
            augmentation_model.add(tf.keras.layers.experimental.preprocessing.RandomContrast(factor = 0.2))

        self.augmentation_model = augmentation_model


    def remove_augmentation(self):

        self.augmentation_model = None