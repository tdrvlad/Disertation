
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


rel_path = os.path.relpath(os.path.dirname(os.path.realpath(__file__)), os.getcwd()) 


DATA_SPLIT = 'val2017'

annotations_file = os.path.join(rel_path, 'annotations/instances_{}.json'.format(DATA_SPLIT))
images_dir = os.path.join(rel_path, DATA_SPLIT)

LABEL_MAP = {
    0: "unlabeled",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic",
    11: "fire",
    12: "street",
    13: "stop",
    14: "parking",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eye",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports",
    38: "kite",
    39: "baseball",
    40: "baseball",
    41: "skateboard",
    42: "surfboard",
    43: "tennis",
    44: "bottle",
    45: "plate",
    46: "wine",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted",
    65: "bed",
    66: "mirror",
    67: "dining",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy",
    89: "hair",
    90: "toothbrush",
    91: "hair",
    92: "banner",
    93: "blanket",
    94: "branch",
    95: "bridge",
    96: "building",
    97: "bush",
    98: "cabinet",
    99: "cage",
    100: "cardboard",
    101: "carpet",
    102: "ceiling",
    103: "ceiling",
    104: "cloth",
    105: "clothes",
    106: "clouds",
    107: "counter",
    108: "cupboard",
    109: "curtain",
    110: "desk",
    111: "dirt",
    112: "door",
    113: "fence",
    114: "floor",
    115: "floor",
    116: "floor",
    117: "floor",
    118: "floor",
    119: "flower",
    120: "fog",
    121: "food",
    122: "fruit",
    123: "furniture",
    124: "grass",
    125: "gravel",
    126: "ground",
    127: "hill",
    128: "house",
    129: "leaves",
    130: "light",
    131: "mat",
    132: "metal",
    133: "mirror",
    134: "moss",
    135: "mountain",
    136: "mud",
    137: "napkin",
    138: "net",
    139: "paper",
    140: "pavement",
    141: "pillow",
    142: "plant",
    143: "plastic",
    144: "platform",
    145: "playingfield",
    146: "railing",
    147: "railroad",
    148: "river",
    149: "road",
    150: "rock",
    151: "roof",
    152: "rug",
    153: "salad",
    154: "sand",
    155: "sea",
    156: "shelf",
    157: "sky",
    158: "skyscraper",
    159: "snow",
    160: "solid",
    161: "stairs",
    162: "stone",
    163: "straw",
    164: "structural",
    165: "table",
    166: "tent",
    167: "textile",
    168: "towel",
    169: "tree",
    170: "vegetable",
    171: "wall",
    172: "wall",
    173: "wall",
    174: "wall",
    175: "wall",
    176: "wall",
    177: "wall",
    178: "water",
    179: "waterdrops",
    180: "window",
    181: "window",
    182: "wood"
}
REVERSE_LABEL_MAP = {v:k for k,v in LABEL_MAP.items()}

def get_image_ids(annotations_file):

    with open(annotations_file) as json_file:
        data = json.load(json_file)

    annotations = data.get('annotations')

    image_ids = set([])
    for annotation in annotations:
        image_ids.add(annotation.get('image_id'))
    
    return list(image_ids)

        
def get_bboxes(annotations_file):

    with open(annotations_file) as json_file:
        data = json.load(json_file)

    bboxes = []
    annotations = data.get('annotations')
    for annotation in annotations:
        bbox_dict = {}
        bbox_dict['bbox'] = annotation.get('bbox')
        bbox_dict['image_id'] = annotation.get('image_id')
        bbox_dict['category_id'] = annotation.get('category_id')
        bboxes.append(bbox_dict)
    
    return bboxes


def get_image_detections(annotations_file, removed_image_ids = []):

    bboxes = get_bboxes(annotations_file)
    image_detections = {}

    for bbox in bboxes:
        image_id = bbox['image_id']
        if image_id not in removed_image_ids:
            if image_detections.get(image_id) is None:
                image_detections[image_id] = {}
            detection_id = '{}_{}'.format(image_id, len(image_detections[image_id]))
            image_detections[image_id][detection_id] = bbox

    return image_detections


def get_image_data(annotations_file, images_dir):

    image_ids = get_image_ids(annotations_file)

    image_data = {}
    removed_image_ids = []
    for image_id in image_ids:
        image_path = '{}.jpg'.format(str(image_id).zfill(12))
        pil_image = Image.open(os.path.join(images_dir, image_path ))

        np_image = np.array(pil_image)
        if len(np_image.shape) == 3:
            image_data[image_id] = np_image
        else:
            removed_image_ids.append(image_id)

    return image_data, removed_image_ids


def crop_detection(np_image, detection, target_size = (224, 224)):

    crop_border = 0.1

    assert len(np_image.shape) == 3, 'Image does not have 3 channels'
    h, w, _ = np_image.shape

    bbox = detection.get('bbox')

    x0 = max(0, int(bbox[0] - crop_border * w))
    y0 = max(0, int(bbox[1] - crop_border * h))
    x1 = min(w, int(bbox[0] + bbox[2] + crop_border * w))
    y1 = min(h, int(bbox[1] + bbox[3] + crop_border * h))
    
    cropped_np_image = np_image[y0:y1, x0:x1]
    pil_cropped_image = Image.fromarray(cropped_np_image)

    return np.array(pil_cropped_image.resize(target_size))


def get_detection_data(image_data, image_detections):

    '''
    detection_data = {}
    if os.path.exists('test_images'):
        shutil.rmtree('test_images')
    os.mkdir('test_images')
    '''

    detection_data = {}

    for image_id, detections in image_detections.items():
        np_image = image_data.get(image_id)
        
        if np_image is not None:
            for detection_id, bbox in detections.items():
            
                detection_crop = crop_detection(np_image, bbox)
                detection_data[detection_id] = detection_crop
                
                #pil_crop = Image.fromarray(detection_crop)
                #pil_crop.save(os.path.join('test_images', '{}.jpg'.format(detection_id)))
    
    return detection_data
          

def get_image_detections_and_detection_data():

    global annotations_file, images_dir
    image_data, removed_image_ids = get_image_data(annotations_file, images_dir)
    image_detections = get_image_detections(annotations_file, removed_image_ids)
    detection_data = get_detection_data(image_data, image_detections)

    return image_detections, detection_data

           
#image_data, removed_image_ids = get_image_data(annotations_file, images_dir)
#image_detections = get_image_detections(annotations_file, removed_image_ids)
#detection_data = get_detection_data(image_data, image_detections)



