
import numpy as np
import random, os, glob, io, json, math, yaml, time, sys
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw
from itertools import combinations 
from datetime import datetime
import shutil
import math
import re

import tensorflow as tf 
import tensorflow_addons as tfa


rel_path = os.path.relpath(os.path.dirname(os.path.realpath(__file__)), os.getcwd()) 

MODELS_DIR = os.path.join(rel_path, 'models')


def generate_projector_files(model_name, data_handler, label_map, no_samples = 2000, max_category_samples = 200, doublehead_model = False):

    data_handler.remove_augmentation()

    x,y = data_handler.create_batch(no_samples)
    labels = [label_map.get(e) for e in list(y[:,0])]
    
    embeddings = predict(model_name, x, doublehead_model = doublehead_model)

    filtered_labels = []
    filtered_embeddings = []
    category_samples = {}

    for i in range(len(embeddings)):
        if category_samples.get(labels[i]) is None:
            category_samples[labels[i]] = 0
        category_samples[labels[i]] += 1
        
        if category_samples[labels[i]] < max_category_samples:
            filtered_labels.append(labels[i])
            filtered_embeddings.append(embeddings[i])
    
    print(category_samples)


        
    genereate_tensorboard_projector_files(filtered_embeddings, filtered_labels, os.path.join(MODELS_DIR, model_name, 'logs') , data_split = 'train')
    

def add_convolutional(encoder_conv_layers, filters, kernel_size, strides = 1, pool_size = None, pool_strides = None, dropout_rate = None, decoder_conv_layers = None):
    
    encoder_conv_layers.append(
        tf.keras.layers.Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = "same",
            activation ='relu'
        )
    )

    encoder_conv_layers.append(tf.keras.layers.BatchNormalization(axis=-1))
    
    if not decoder_conv_layers is None:
        decoder_conv_layers.append(tf.keras.layers.BatchNormalization(axis=-1))
    
    if not pool_size is None and not pool_strides is None:
        transpose_strides = strides * pool_strides

        encoder_conv_layers.append(
            tf.keras.layers.MaxPooling2D(
                pool_size = pool_size,
                padding = "same",
                strides = pool_strides,
            ))

    else:
        transpose_strides = strides

    if not decoder_conv_layers is None:
        decoder_conv_layers.append(
            tf.keras.layers.Conv2DTranspose(
                filters = filters,
                kernel_size = kernel_size,
                strides = transpose_strides,
                padding = "same",
                activation = 'relu'
            )
        )

    if not dropout_rate is None:
        
        encoder_conv_layers.append(
            tf.keras.layers.Dropout(rate = dropout_rate)
        )


def add_dense_layer(no_units, encoder_dense_layers, decoder_dense_layers = None, dropout_rate = 0, add_normalization = False, activation = 'relu'):

    
    encoder_dense_layers.append(tf.keras.layers.Dense(
            units = no_units,
            activation=activation
        )
    )
    
    if dropout_rate:
        encoder_dense_layers.append(
            tf.keras.layers.Dropout(rate = dropout_rate)
        )

    if add_normalization:
        encoder_dense_layers.append(tf.keras.layers.BatchNormalization(axis=-1))

    if decoder_dense_layers is not None:
        decoder_dense_layers.append(tf.keras.layers.Dense(
                units = no_units,
                activation='relu'
            )
        )



def create_doublehead_model(input_shape = (32,32,3), embedding_size = 128, hidden_layer_neurons = 0, model_name = 'doublehead_model'):

    encoder_conv_layers = []
    decoder_conv_layers = []
    encoder_dense_layers = []
    decoder_dense_layers = []
    

    'Defining layers'
    add_convolutional(encoder_conv_layers, decoder_conv_layers = decoder_conv_layers, filters = 64, kernel_size = 3)
    add_convolutional(encoder_conv_layers, decoder_conv_layers = decoder_conv_layers, filters = 64, kernel_size = 3, pool_size = 2, pool_strides = 2)
    add_convolutional(encoder_conv_layers, decoder_conv_layers = decoder_conv_layers, filters = 128, kernel_size = 3)
    add_convolutional(encoder_conv_layers, decoder_conv_layers = decoder_conv_layers, filters = 128, kernel_size = 3, pool_size = 2, pool_strides = 2)
    add_convolutional(encoder_conv_layers, decoder_conv_layers = decoder_conv_layers, filters = 256, kernel_size = 3)
    add_convolutional(encoder_conv_layers, decoder_conv_layers = decoder_conv_layers, filters = 256, kernel_size = 3, pool_size = 2, pool_strides = 2)
    add_convolutional(encoder_conv_layers, decoder_conv_layers = decoder_conv_layers, filters = 512, kernel_size = 3)
    add_convolutional(encoder_conv_layers, decoder_conv_layers = decoder_conv_layers, filters = 512, kernel_size = 3, pool_size = 2, pool_strides = 2)
    
    if hidden_layer_neurons:
        add_dense_layer(hidden_layer_neurons, encoder_dense_layers, decoder_dense_layers = decoder_dense_layers, dropout_rate = 0.2)
    add_dense_layer(embedding_size, encoder_dense_layers, decoder_dense_layers = None, activation = None)
    
    model_input = tf.keras.Input(shape = (input_shape[1], input_shape[0], 3))

    'Building Encoder'

    encoder_input = model_input
    obj = encoder_input

    for layer in encoder_conv_layers:
        obj = layer(obj)

    conv_shape = obj.shape[1:]

    obj = tf.keras.layers.Flatten()(obj)
    flatten_shape = obj.shape[1]

    for layer in encoder_dense_layers:
        obj = layer(obj)
    
    encoder_output = obj
    
    encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")
    encoder.summary()


    'Building Decoder'

    decoder_conv_layers.reverse()
    decoder_dense_layers.reverse()

    decoder_input = tf.keras.Input(shape = encoder_output.shape[1:])
    obj = decoder_input
    
    for layer in decoder_dense_layers:
        obj = layer(obj)

    obj = tf.keras.layers.Dense(units = int(flatten_shape), activation='relu')(obj)
    obj = tf.keras.layers.Reshape(conv_shape)(obj)

    for layer in decoder_conv_layers:
        obj = layer(obj)

    decoder_output = tf.keras.layers.Conv2DTranspose(3, 3, activation="relu", padding="same")(obj)
    decoder = tf.keras.Model(decoder_input, decoder_output, name="decoder")
    decoder.summary()

    
    'Building end-to-end Model'

    embedding_obj = encoder(model_input)
    reconstruction_output = decoder(embedding_obj)

    model = tf.keras.Model(model_input, [embedding_obj, reconstruction_output], name = 'model')
    model.summary()
    model.save(os.path.join(MODELS_DIR, model_name))


def create_model(input_shape = (224,224,3), embedding_size = 256, intermediate_layer_size = 512, model_name = 'final_model_ImageNet'):

    input_obj = tf.keras.layers.Input(input_shape)

    backbone = tf.keras.applications.MobileNet(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"
    )

    obj = backbone(input_obj)
    obj = tf.keras.layers.Flatten()(obj)

    if intermediate_layer_size:
        intermediate_layer = tf.keras.layers.Dense(intermediate_layer_size, activation='relu')
        obj = intermediate_layer(obj)
    
    embedding_layer = tf.keras.layers.Dense(embedding_size , activation=None)
    output_obj = embedding_layer(obj)

    model = tf.keras.Model(input_obj, output_obj)

    model.summary()
    model.save(os.path.join(MODELS_DIR, model_name))



def train_model(model_name, data_handler, new_model_name = None, steps_per_epoch = None, no_epochs = 10, learning_rate = 0.01, entity_loss_weight = 1, context_loss_weight = 1, freeze_backbone = True, doublehead = False, reconstruction_loss_weight = 1, apply_imagenet_preprocessing = False, skip_verification = True):

    batch_size = 32
    data_generator = data_handler.batch_generator(batch_size = batch_size, autoencoder_label = doublehead, apply_imagenet_preprocessing = apply_imagenet_preprocessing)

    if steps_per_epoch is None:
        steps_per_epoch = int(data_handler.get_no_items() / batch_size)

    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, model_name), custom_objects={'EntityContextTripletLoss': EntityContextTripletLoss()})

    if new_model_name is None:
        new_model_name = 'training_{}_{}'.format(model_name, time.time())
    else:
        new_model_name = '{}_{}'.format(new_model_name, time.time())

    new_model_dir = os.path.join(MODELS_DIR, new_model_name)
    
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    entity_context_triplet_loss = EntityContextTripletLoss(entity_margin = 1, entity_loss_weight = entity_loss_weight, context_margin = 1, context_loss_weight = context_loss_weight)
    
    if doublehead:
        reconstruction_loss = tf.keras.losses.MeanSquaredError()
        loss = [entity_context_triplet_loss, reconstruction_loss]
        loss_weights = [1, reconstruction_loss_weight]
    else:
        loss = entity_context_triplet_loss  
        loss_weights = None
        
    model.compile(
        optimizer=opt,
        loss = loss,
        loss_weights = loss_weights
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(new_model_dir, 'logs'))
    save_callback = tf.keras.callbacks.ModelCheckpoint(new_model_dir, save_freq="epoch")

    if freeze_backbone:
        model.layers[3].trainable = False

    model.summary()

    if not skip_verification:
        input("Press Enter to continue...")

    model.fit(
        data_generator,
        epochs=no_epochs,
        callbacks = [tensorboard_callback, save_callback],
        steps_per_epoch = steps_per_epoch,
    )
    
    model.save(new_model_dir)


def save_model_summary(model_name):

    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, model_name), custom_objects={'EntityContextTripletLoss': EntityContextTripletLoss()})

    from contextlib import redirect_stdout

    summary_file = os.path.join(MODELS_DIR, model_name, 'model_summary.txt')
    plot_file = os.path.join(MODELS_DIR, model_name, 'model_plot.png')

    '''
    with open(summary_file, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    '''

    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True)



def predict(model_name, np_image_batch, doublehead_model = False):

    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, model_name), custom_objects={'EntityContextTripletLoss': EntityContextTripletLoss()})

    batch_size = 64
    no_complete_batches = int(len(np_image_batch) / batch_size)
    embeddings = []

    for i in range(no_complete_batches):
        pred = model.predict(np.stack(np_image_batch[i * batch_size: (i+1) * batch_size]))
        if doublehead_model:  
            pred = pred[0]

        embeddings.extend(list(pred))

    pred = model.predict(np.stack(np_image_batch[no_complete_batches * batch_size:]))

    if doublehead_model:  
        pred = pred[0]
    
    embeddings.extend(list(pred))

    return embeddings


def add_mobilenet_preprocessing_to_model(model_name):

    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, model_name), custom_objects={'EntityContextTripletLoss': EntityContextTripletLoss()})
    input_obj = tf.keras.layers.Input(model.layers[0].get_input_shape_at(0)[1:])

    obj = tf.keras.applications.mobilenet.preprocess_input(input_obj)
    for layer in model.layers[1:]:
        obj = layer(obj)

    new_model = tf.keras.Model(input_obj, obj)
    new_model.summary()

    new_model.save(os.path.join(MODELS_DIR, model_name + '_with_preprocessing'))
    


class EntityContextTripletLoss(tf.keras.losses.Loss):

    def __init__(self, entity_margin = 1, entity_loss_weight = 1, context_margin = 1, context_loss_weight = 1, reduction=tf.keras.losses.Reduction.AUTO, name='entity_context_triplet_loss'):
        
        super().__init__(reduction=reduction, name=name)
        
        self.entity_loss = tfa.losses.TripletSemiHardLoss(margin = float(entity_margin))
        self.entity_loss_weight = entity_loss_weight

        self.context_loss = tfa.losses.TripletSemiHardLoss(margin = float(context_margin))
        self.context_loss_weight = context_loss_weight


    def call(self, y_true, y_pred):

        y_entity = y_true[:,0]
        y_context = y_true[:,1]

        y_pred = tf.cast(y_pred, tf.float32)
   
        entity_loss_value = self.entity_loss(y_entity, y_pred)

        if self.context_loss_weight:
            context_loss_value = self.context_loss(y_context, y_pred)
        else:
            context_loss_value = 0

        return self.entity_loss_weight * entity_loss_value + self.context_loss_weight * context_loss_value


def genereate_tensorboard_projector_files(embeddings, labels, directory, data_split = 'train'):

    if not os.path.exists(directory):
        os.mkdir(directory)

    np.savetxt(os.path.join(directory, "vecs_{}.tsv".format(data_split)), embeddings, fmt='%.1f',  delimiter='\t')
    out_m = io.open(os.path.join(directory, 'meta_{}.tsv'.format(data_split)), 'w', encoding='utf-8')
    [out_m.write(str(x) + "\n") for x in labels]
    out_m.close()


if __name__ =='__main__':
    save_model_summary('final_model_ImageNet')