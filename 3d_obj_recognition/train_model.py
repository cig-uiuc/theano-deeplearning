from keras.models import Sequential
from keras.models import Graph
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from keras.models import model_from_json

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys, glob, pdb

import EitelModel as eitel
import data_processor as dtp

SAVE_MODEL = True
DICT = 'lists/dictionary.txt'
MODEL_LOC = 'models/'
RGB_MODEL_NAME = 'rgb_stream'
DEP_MODEL_NAME = 'dep_stream'
FUS_MODEL_NAME = 'fusion_model'
STRUCT_EXT = '.json'
WEIGHT_EXT = '.h5'


def train_model(model, x_train, y_train, b_size):
    model.fit(x_train, y_train, nb_epoch=1, batch_size=b_size)
    return model


def main():
    # load paths-----------------------------------------------------------------
    all_data_path = dtp.get_all_data_path()
    categories = open(DICT, 'r+').read().splitlines()

    batch_size = 50
    nb_samples = len(all_data_path)
    total_batch = int(np.ceil(1.0*nb_samples/batch_size))

    # generate stream model------------------------------------------------------
    if not os.path.isfile(MODEL_LOC+RGB_MODEL_NAME+STRUCT_EXT) or not os.path.isfile(MODEL_LOC+RGB_MODEL_NAME+WEIGHT_EXT):
        print 'Generating RGB stream models...'
        rgb_stream = eitel.create_single_stream(len(categories))
        plot(rgb_stream, 'stream_model.png')

        print 'Training RGB stream models...'
        for batch_id in range(0, nb_samples, batch_size):
            print 'Training batch', batch_id/batch_size+1, 'of', total_batch, '...'

            batch = all_data_path[batch_id:batch_id+batch_size]
            rgb_x_train, _, y_train = dtp.get_data(batch,categories)
        
            rgb_stream = train_model(rgb_stream, rgb_x_train, y_train, len(y_train))
    
        if SAVE_MODEL:
            json_str = rgb_stream.to_json()
            open(MODEL_LOC+RGB_MODEL_NAME+STRUCT_EXT, 'w').write(json_str)
            rgb_stream.save_weights(MODEL_LOC+RGB_MODEL_NAME+WEIGHT_EXT, overwrite=True)
            del rgb_stream
    else:
        print 'RGB model exists...'

    # generate stream model-----------------------------------------------------
    if not os.path.isfile(MODEL_LOC+DEP_MODEL_NAME+STRUCT_EXT) or not os.path.isfile(MODEL_LOC+DEP_MODEL_NAME+WEIGHT_EXT):
        print 'Generating depth stream models...'
        dep_stream = eitel.create_single_stream(len(categories))

        print 'Training depth stream models...'
        for batch_id in range(0, nb_samples, batch_size):
            print 'Training batch', batch_id/batch_size+1, 'of', total_batch, '...'

            batch = all_data_path[batch_id:batch_id+batch_size]
            _, dep_x_train, y_train = dtp.get_data(batch,categories)
        
            dep_stream = train_model(dep_stream, dep_x_train, y_train, len(y_train))
    
        if SAVE_MODEL:
            json_str = dep_stream.to_json()
            open(MODEL_LOC+DEP_MODEL_NAME+STRUCT_EXT, 'w').write(json_str)
            dep_stream.save_weights(MODEL_LOC+DEP_MODEL_NAME+WEIGHT_EXT, overwrite=True)
            del dep_stream
    else:
        print 'Depth model exitsts...'


    # reload the model weights----------------------------------------------------
    print 'Loading weights...'
    rgb_stream = model_from_json(open(MODEL_LOC+RGB_MODEL_NAME+STRUCT_EXT).read())
    rgb_stream.load_weights(MODEL_LOC+RGB_MODEL_NAME+WEIGHT_EXT)
    rgb_layers = rgb_stream.layers
    del rgb_stream

    dep_stream = model_from_json(open(MODEL_LOC+DEP_MODEL_NAME+STRUCT_EXT).read())
    dep_stream.load_weights(MODEL_LOC+DEP_MODEL_NAME+WEIGHT_EXT)
    dep_layers = dep_stream.layers
    del dep_stream

    # fuse stream model-----------------------------------------------------------
    print 'Fusing stream models...'
    fusion_model = eitel.create_model_merge(rgb_layers, dep_layers, len(categories))
    del rgb_layers
    del dep_layers
    plot(fusion_model, 'fusion_model.png')

    # reduce batch size to save memory
    batch_size = 10
    nb_samples = len(all_data_path)
    total_batch = int(np.ceil(1.0*nb_samples/batch_size))

    # train fusion model
    for batch_id in range(0, nb_samples, batch_size):
        print 'Fusing batch', batch_id/batch_size+1, 'of', total_batch, '...'

        batch = all_data_path[batch_id:batch_id+batch_size]
        rgb_x_train, dep_x_train, y_train = dtp.get_data(batch, categories)

        fusion_model = train_model(fusion_model, [rgb_x_train, dep_x_train], y_train, len(y_train))

    if SAVE_MODEL:
        json_str = fusion_model.to_json()
        open(MODEL_LOC+FUS_MODEL_NAME+STRUCT_EXT, 'w').write(json_str)
        fusion_model.save_weights(MODEL_LOC+FUS_MODEL_NAME+WEIGHT_EXT, overwrite=True)


if __name__ == '__main__':
    main()
