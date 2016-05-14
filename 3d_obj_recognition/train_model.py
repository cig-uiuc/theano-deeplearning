from keras.models import Graph
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from keras.models import model_from_json


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys, glob, pdb
import time
import progressbar
from random import shuffle

import EitelModel as eitel
import data_processor as dtp


# Constants=========================================================================
SAVE_MODEL = True

LIST_LOC = './lists/'
DATA_LOC = '/media/data/washington_dataset/fullset/cropped/'

#DICT       = './lists/dictionary_small.txt'
#MODEL_LOC  = './models/small/'
#TRAIN_LIST = 'train_list_small.txt'
#EVAL_LIST  = 'eval_list_small.txt'
DICT       = './lists/dictionary_full.txt'
MODEL_LOC  = './models/full/'
TRAIN_LIST = 'train_list_full.txt'
EVAL_LIST  = 'eval_list_full.txt'

PRETRAINED_LOC = './pretrained/'
PRETRAINED_NAME = 'keras_alexnet'

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


def load_model(loc, name):
    '''
    if use_marcbs:
        model = marcbs_model_from_json(open(loc+name+STRUCT_EXT).read())
        model.load_weights(loc+name+WEIGHT_EXT)
    else:
        model = model_from_json(open(loc+name+STRUCT_EXT).read())
        model.load_weights(loc+name+WEIGHT_EXT)
    '''

    model = model_from_json(open(loc+name+STRUCT_EXT).read())
    model.load_weights(loc+name+WEIGHT_EXT)
    return model


def switch_input(rgb_x, dep_x, mode):
    if mode == 0:
        x = rgb_x
    elif mode == 1:
        x = dep_x
    elif mode == 2:
        x = [rgb_x, dep_x]
    return x


def gen_graph_input_params(rgb_x, dep_x, mode):
    if mode==0:
        params = {'input_rgb':rgb_x}
    elif mode==1:
        params = {'input_dep':dep_x}
    elif mode==2:
        params = {'input_rgb':rgb_x, 'input_dep':dep_x}

    return params


def compute_accuracy(y_pred, y_true):
    N = y_pred.shape[0]
    count = 0.0
    for i in range(N):
        if y_pred[i].argmax() == y_true[i].argmax():
            count += 1.0
    acc_on_batch = count/N
    return count, acc_on_batch


def train_model(model, mode, batch_size, train_samples, eval_samples, classes):
    '''
    mode = 0: use only rgb
    mode = 1: use only dep
    mode = 2: use both rgb and dep
    '''
    nb_epoch = 1000
    patience = 5
    min_loss = sys.maxint
    max_accuracy = 0
    nb_static_epoch = 0
    epsilon = 0.0001

    #eval = train
    #eval_samples = train_samples

    nb_train_samples = len(train_samples)
    nb_eval_samples  = len(eval_samples)
    nb_train_batches = int(np.ceil(1.0*nb_train_samples/batch_size))
    nb_eval_batches  = int(np.ceil(1.0*nb_eval_samples/batch_size))

    train_bar = progressbar.ProgressBar(maxval=nb_train_samples, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    eval_bar = progressbar.ProgressBar(maxval=nb_eval_samples, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])


    if mode==0:
        f = open('rgb_train.log', 'w', 0)
    elif mode==1:
        f = open('dep_train.log', 'w', 0)
    elif mode==2:
        f = open('fus_train.log', 'w', 0)


    for epoch in range(nb_epoch):
        print 'Running epoch '+ str(epoch+1) + '/'+ str(nb_epoch) + '...'
        f.write('Running epoch '+ str(epoch+1) + '/'+ str(nb_epoch) + '...\n')

        # training---------------------------------------------------------
        print 'Training phase'
        f.write('Training phase------------------------------------------------\n')

        train_bar.start()
        start_time = time.time()
        shuffle(train_samples)
        for batch_id in range(0, nb_train_samples, batch_size):
            train_bar.update(batch_id)
            batch = train_samples[batch_id : batch_id+batch_size]
            rgb_x, dep_x, y = dtp.load_data(batch, classes, DATA_LOC)
            params = gen_graph_input_params(rgb_x, dep_x, mode)
            
            # find loss
            params['output'] = y
            loss = model.train_on_batch(params)

            # find accuracy
            del params['output']
            pred = model.predict(params)
            acc_count, acc_on_batch = compute_accuracy(pred.get('output'), y)

            f.write(str(loss[0]) + ' --- ' + str(acc_on_batch) + '\n')
        train_bar.finish()
        elapsed_time = time.time()-start_time
        print 'Elapsed time: ' + str(elapsed_time) + 's'


        # evaluating-------------------------------------------------------
        print 'Evaluation phase'
        f.write('Evaluation phase-----------------------------------------------\n')
        avg_loss = 0
        avg_accuracy = 0
        eval_bar.start()

        for batch_id in range(0, nb_eval_samples, batch_size):
            eval_bar.update(batch_id)
            batch = eval_samples[batch_id : batch_id+batch_size]
            rgb_x, dep_x, y = dtp.load_data(batch, classes, DATA_LOC)
            params = gen_graph_input_params(rgb_x, dep_x, mode)

            # find losss
            params['output'] = y
            loss = model.test_on_batch(params)

            # find accuracy
            del params['output']
            pred = model.predict(params)
            acc_count, acc_on_batch = compute_accuracy(pred.get('output'), y)

            f.write(str(loss[0]) + ' --- ' + str(acc_on_batch) + '\n')

            # accumulate loss and accuracy
            avg_loss += loss[0]
            avg_accuracy += acc_count
        eval_bar.finish()


        # check accuracy---------------------------------------------------
        avg_loss /= nb_eval_batches
        avg_accuracy /= nb_eval_samples
        improvement = avg_accuracy - max_accuracy
        print 'Average loss: '+str(avg_loss)+' - Average accuracy: '+str(avg_accuracy)+' - Improvement: '+str(improvement)+'\n' 
        f.write('Average loss: '+str(avg_loss)+' - Average accuracy: '+str(avg_accuracy)+' - Improvement: '+str(improvement)+'\n\n')
        #print 'Accuracy: '+str(avg_accuracy)+' - Improvement: '+str(improvement)+'\n'
        if max_accuracy != 0:
            improvement /= max_accuracy

        if improvement > epsilon:
            #min_loss = avg_loss
            max_accuracy = avg_accuracy
            nb_static_epoch = 0
        else:
            nb_static_epoch += 1
            if nb_static_epoch >= patience:
                print 'Accuracy does not imrpove. Early stopping...\n'
                f.write('Accuracy does not imrpove. Early stopping...\n\n')
                break

    f.close()
    return model


def main():
    # init-----------------------------------------------------------------
    train_samples = dtp.sample_paths_from_list(DATA_LOC, LIST_LOC+TRAIN_LIST)
    eval_samples  = dtp.sample_paths_from_list(DATA_LOC, LIST_LOC+EVAL_LIST)
    classes = open(DICT, 'r+').read().splitlines()
    nb_classes = len(classes)
    batch_size = 180

    # load pretrained models----------------------------------------------------
    pretrained = load_model(PRETRAINED_LOC, PRETRAINED_NAME)

    # generate RGB stream model------------------------------------------------------
    if not model_exist(RGB_MODEL_NAME):
        print 'Training RGB stream model...\n'
        rgb_stream = eitel.create_single_stream(nb_classes, pretrained, tag='_rgb')
        plot(rgb_stream, 'stream_model.png')
        rgb_stream = train_model(rgb_stream, 0, batch_size, train_samples, eval_samples, classes)
        
        if SAVE_MODEL:
            save_model(rgb_stream, RGB_MODEL_NAME)
            del rgb_stream
    else:
        print 'RGB stream model already exists...'

    # generate RGB stream model------------------------------------------------------
    if not model_exist(DEP_MODEL_NAME):
        print 'Training depth stream model...\n'
        dep_stream = eitel.create_single_stream(nb_classes, pretrained, tag='_dep')
        dep_stream = train_model(dep_stream, 1, batch_size, train_samples, eval_samples, classes)
        
        if SAVE_MODEL:
            save_model(dep_stream, DEP_MODEL_NAME)
            del dep_stream
    else:
        print 'Depth stream model already exists...'

    # reload the model weights----------------------------------------------------
    print 'Loading weights...'
    rgb_stream = load_model(MODEL_LOC, RGB_MODEL_NAME)
    dep_stream = load_model(MODEL_LOC, DEP_MODEL_NAME)

    # fusion model-----------------------------------------------------------
    print 'Fusing stream models...'
    fusion_model = eitel.create_model_merge(rgb_stream, dep_stream, nb_classes)
    del rgb_stream
    del dep_stream
    plot(fusion_model, 'fusion_model.png')

    batch_size=500
    fusion_model = train_model(fusion_model, 2, batch_size, train_samples, eval_samples, classes)
    if SAVE_MODEL:
        save_model(fusion_model, FUS_MODEL_NAME)
    

if __name__ == '__main__':
    main()
