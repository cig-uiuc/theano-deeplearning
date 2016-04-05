import theano

from keras.models import Sequential
from keras.models import Graph
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys, glob, pdb

import EitelModel as eitel
import data_processor as dtp

SAVE_MODEL = True
DICT = 'dictionary.txt'

'''
def create_model(nb_class):
    model = Sequential()

    # conv-1 layer
    conv_1 = Convolution2D(96, 3, 3, border_mode='valid', input_shape=(3,IMG_S,IMG_S))
    model.add(conv_1)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-2 layer
    conv_2 = Convolution2D(256, 3, 3, border_mode='valid')
    model.add(conv_2)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-3 layer
    conv_3 = Convolution2D(384, 3, 3, border_mode='valid')
    model.add(conv_3)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-4 layer
    conv_4 = Convolution2D(384, 3, 3, border_mode='valid')
    model.add(conv_4)
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # conv-5 layer
    conv_5 = Convolution2D(256, 3, 3, border_mode='valid')
    model.add(conv_5)
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    #print conv_5.output_shape # = (256 x 22 x 22)

    # fc6 layer
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # fc7 layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # dummy output layer ==> need to change when fusing the RGB and D models
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    
    # compile model
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    #plot(model, 'train_model.png')

    return model
'''


def train_model(model, x_train, y_train, b_size):
    model.fit(x_train, y_train, nb_epoch=1, batch_size=b_size)
    return model


def main():
    # use GPU--------------------------------------------------------------------
    #theano.config.device = 'gpu'
    #theano.config.floatX = 'float32'

    # load paths-----------------------------------------------------------------
    all_data_path = dtp.get_all_data_path()
    categories = open(DICT, 'r+').read().splitlines()


    # generate stream models-----------------------------------------------------
    print 'Generating stream models...'
    rgb_stream = eitel.create_single_stream(len(categories))
    dep_stream = eitel.create_single_stream(len(categories))
    plot(rgb_stream, 'stream_model.png')


    # train stream model (by batch)----------------------------------------------
    print 'Training stream models...'

    batch_size = 100
    nb_samples = len(all_data_path)
    total_batch = int(np.ceil(1.0*nb_samples/batch_size))

    '''# test code
    batch_size = 1
    nb_samples = batch_size # for testing => run 1 batch only
    total_batch = int(np.ceil(1.0*nb_samples/batch_size))'''

    for batch_id in range(0, nb_samples, batch_size):
        print 'Training batch', batch_id/batch_size+1, 'of', total_batch, '...'

        batch = all_data_path[batch_id:batch_id+batch_size]
        rgb_x_train, dep_x_train, y_train = dtp.get_data(batch,categories)
        
        rgb_stream = train_model(rgb_stream, rgb_x_train, y_train, len(y_train))
        dep_stream = train_model(dep_stream, dep_x_train, y_train, len(y_train))
    
    if SAVE_MODEL:
        rgb_stream.save_weights('rgb_stream.h5', overwrite=TRUE)
        dep_stream.save_weights('dep_strean.h5', overwrite=TRUE)


    # fuse stream models---------------------------------------------------------
    print 'Fusing stream models...'
    fusion_model = eitel.create_model_merge(rgb_stream, dep_stream, len(categories))
    plot(fusion_model, 'fusion_model.png')
    for batch_id in range(0, nb_samples, batch_size):
        print 'Fusing batch', batch_id/batch_size+1, 'of', total_batch, '...'

        batch = all_data_path[batch_id:batch_id+batch_size]
        rgb_x_train, dep_x_train, y_train = dtp.get_data(batch, categories)

        fusion_model = train_model(fusion_model, [rgb_x_train, dep_x_train], y_train, len(y_train))

    if SAVE_MODEL:
        fusion_model.save_weights('fusion_model.h5', overwrite=TRUE)


if __name__ == '__main__':
    main()
