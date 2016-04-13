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
import progressbar

import EitelModel as eitel
import data_processor as dtp


# Constants=========================================================================
SAVE_MODEL = True

LIST_LOC = 'lists/'
DATA_LOC = '/media/data/washington_dataset/fullset/cropped/'

DICT       = 'lists/dictionary_small.txt'
MODEL_LOC  = 'models/small/'
TRAIN_LIST = 'train_list_small.txt'
EVAL_LIST  = 'eval_list_small.txt'
TEST_LIST  = 'test_list_small.txt'

RGB_MODEL_NAME = 'rgb_stream'
DEP_MODEL_NAME = 'dep_stream'
FUS_MODEL_NAME = 'fusion_model'
STRUCT_EXT     = '.json'
WEIGHT_EXT     = '.h5'



# Functions=========================================================================
def model_exist(model_name):
    prefix = MODEL_LOC+model_name
    if os.path.isfile(prefix+STRUCT_EXT) and os.path.isfile(prefix+WEIGHT_EXT):
        return True
    return False


def save_model(model, name):
    json_str = model.to_json()
    open(MODEL_LOC+name+STRUCT_EXT, 'w').write(json_str)
    model.save_weights(MODEL_LOC+name+WEIGHT_EXT, overwrite=True) 


def train_model(model, mode, batch_size, train_samples, eval_samples, classes):
    '''
    mode = 0: use only rgb
    mode = 1: use only dep
    mode = 2: use both rgb and dep
    '''

    nb_epoch = 1000
    patience = 3
    min_loss = sys.maxint
    nb_static_epoch = 0
    epsilon = 1e-5

    nb_train_samples = len(train_samples)
    nb_eval_samples  = len(eval_samples)
    nb_train_batches = int(np.ceil(1.0*nb_train_samples/batch_size))
    nb_eval_batches  = int(np.ceil(1.0*nb_eval_samples/batch_size))

    train_bar = progressbar.ProgressBar(maxval=nb_train_samples, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    eval_bar = progressbar.ProgressBar(maxval=nb_eval_samples, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])


    for epoch in range(nb_epoch):
        print 'Running epoch '+ str(epoch+1) + '/'+ str(nb_epoch) + '...'

        # training---------------------------------------------------------
        print 'Training phase'
        train_bar.start()
        for batch_id in range(0, nb_train_samples, batch_size):
            train_bar.update(batch_id)
            batch = train_samples[batch_id : batch_id+batch_size]
            rgb_x, dep_x, y = dtp.load_data(batch, classes, DATA_LOC)

            if mode == 0:
                model.train_on_batch(rgb_x, y)
            elif mode == 1:
                model.train_on_batch(dep_x, y)
            elif mode == 2:
                model.train_on_batch([rgb_x, dep_x], y)
        train_bar.finish()


        # evaluating-------------------------------------------------------
        print 'Evaluation phase'
        avg_loss = 0
        eval_bar.start()
        for batch_id in range(0, nb_eval_samples, batch_size):
            eval_bar.update(batch_id)
            batch = eval_samples[batch_id : batch_id+batch_size]
            rgb_x, dep_x, y = dtp.load_data(batch, classes, DATA_LOC)

            if mode == 0:
                loss = model.test_on_batch(rgb_x, y)
            elif mode == 1:
                loss = model.test_on_batch(dep_x, y)
            elif mode == 2:
                loss = model.test_on_batch([rgb_x, dep_x], y)
            avg_loss += loss[0]
        eval_bar.finish()


        # check average loss-----------------------------------------------
        avg_loss /= nb_eval_batches
        print 'Average loss: '+str(avg_loss)+'\n'

        if abs(avg_loss - min_loss) > epsilon:
            min_loss = avg_loss
            nb_static_epoch = 0
        else:
            nb_static_epoch += 1
            if nb_static_epoch >= patience:
                print 'Model is not imrpoved. Early stopping...\n'
                break

    return model


def main():
    # init-----------------------------------------------------------------
    train_samples = dtp.sample_paths_from_list(DATA_LOC, LIST_LOC+TRAIN_LIST)
    eval_samples  = dtp.sample_paths_from_list(DATA_LOC, LIST_LOC+EVAL_LIST)
    test_samples  = dtp.sample_paths_from_list(DATA_LOC, LIST_LOC+TEST_LIST)
    classes = open(DICT, 'r+').read().splitlines()
    nb_classes = len(classes)
    batch_size = 10

    # generate RGB stream model------------------------------------------------------
    if not model_exist(RGB_MODEL_NAME):
        print 'Training RGB stream model...\n'
        rgb_stream = eitel.create_single_stream(nb_classes)
        # plot(rgb_stream, 'stream_model.png')
        rgb_stream = train_model(rgb_stream, 0, batch_size, train_samples, eval_samples, classes)
        
        if SAVE_MODEL:
            save_model(rgb_stream, RGB_MODEL_NAME)
            del rgb_stream
    else:
        print 'RGB stream model already exists...'

    # generate RGB stream model------------------------------------------------------
    if not model_exist(DEP_MODEL_NAME):
        print 'Training depth stream model...\n'
        dep_stream = eitel.create_single_stream(nb_classes)
        # plot(rgb_stream, 'stream_model.png')
        dep_stream = train_model(dep_stream, 1, batch_size, train_samples, eval_samples, classes)
        
        if SAVE_MODEL:
            save_model(dep_stream, DEP_MODEL_NAME)
            del dep_stream
    else:
        print 'Depth stream model already exists...'

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

    # fusion model-----------------------------------------------------------
    print 'Fusing stream models...'
    fusion_model = eitel.create_model_merge(rgb_layers, dep_layers, nb_classes)
    del rgb_layers
    del dep_layers
    #plot(fusion_model, 'fusion_model.png')

    fusion_model = train_model(fusion_model, 2, batch_size, train_samples, eval_samples, classes)
    if SAVE_MODEL:
        save_model(fusion_model, FUS_MODEL_NAME)
    

if __name__ == '__main__':
    main()
