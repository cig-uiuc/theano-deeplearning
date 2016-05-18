'''
Train fusion model using our own proposed architecture.
RGB and Depth stream models must be prepared first by running file "train_model.py".
'''
from keras.models import model_from_json
from keras.models import Graph
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad
import theano

import data_processor as dtp
import numpy as np
import progressbar, time
from random import shuffle
import pdb


DATA_LOC = '/media/data/washington_dataset/fullset/cropped/'
LIST_LOC = './lists/'
LOG_LOC  = './log/'

MODEL_LOC = './models/full/'
TRAIN_LIST = 'train_list_full.txt'
EVAL_LIST = 'eval_list_full.txt'
TEST_LIST = 'test_list_full.txt'
DICT = './lists/dictionary_full.txt'

RGB_MODEL_NAME = 'rgb_stream'
DEP_MODEL_NAME = 'dep_stream'
FUS_MODEL_NAME = 'fus_model_fc6'
STRUCT_EXT = '.json'
WEIGHT_EXT = '.h5'


def load_model(loc, name):
    model = model_from_json(open(loc+name+STRUCT_EXT).read())
    model.load_weights(loc+name+WEIGHT_EXT)
    return model


def save_model(model, name):
    json_str = model.to_json()
    open(MODEL_LOC+name+STRUCT_EXT, 'w').write(json_str)
    model.save_weights(MODEL_LOC+name+WEIGHT_EXT, overwrite=True)


def create_fusion_model(nb_classes):
    model = Graph()

    # fc6 (from relu)
    model.add_input(name='input_fus', input_shape=(8192,)) # 4096 from rgb + 4096 from depth (merged beforehand)
    model.add_node(Activation('relu'), name='relu6_fus', input='input_fus')
    model.add_node(Dropout(0.5), name='drop6_fus', input='relu6_fus')

    # fc7
    model.add_node(Dense(4096), name='fc7_fus', input='drop6_fus')
    model.add_node(Activation('relu'), name='relu7_fus', input='fc7_fus')
    model.add_node(Dropout(0.5), name='drop7_fus', input='relu7_fus')

    # fc8 - classifier
    model.add_node(Dense(nb_classes), name='fc8_fus', input='drop7_fus')
    model.add_node(Activation('softmax'), name='softmax8_fus', input='fc8_fus')
    model.add_output(name='output', input='softmax8_fus')

    sgd = SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(loss={'output':'categorical_crossentropy'}, optimizer=sgd)

    model.summary()
    return model


def compute_accuracy(y_pred, y_true):
    N = y_pred.shape[0]
    count = 0.0
    for i in range(N):
        if y_pred[i].argmax() == y_true[i].argmax():
            count += 1.0
    acc_on_batch = count/N
    return count, acc_on_batch


def train_model(model, rgb_func, dep_func, train_samples, eval_samples, classes):
    batch_size = 500
    nb_epoch = 1000
    patience = 5
    nb_static_epoch = 0
    epsilon = 1e-4
    max_accuracy = 0

    nb_train_samples = len(train_samples)
    nb_eval_samples  = len(eval_samples)
    nb_train_batches = int(np.ceil(1.0*nb_train_samples/batch_size))
    nb_eval_batches  = int(np.ceil(1.0*nb_eval_samples/batch_size))

    progress_widget = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
    train_bar = progressbar.ProgressBar(maxval=nb_train_samples, widgets=progress_widget)
    eval_bar  = progressbar.ProgressBar(maxval=nb_eval_samples, widgets=progress_widget)

    f = open(LOG_LOC+'fus_fc6_train.log', 'w', 0)


    for epoch in range(nb_epoch):
        print 'Running epoch '+ str(epoch+1) + '/'+ str(nb_epoch) + '...'
        f.write('Running epoch '+ str(epoch+1) + '/'+ str(nb_epoch) + '...\n')

        # training------------------------------------------------------------------------------------
        print 'Training phase'
        f.write('Training phase--------------------------------------------------------------------\n')

        train_bar.start()
        start_time = time.time()
        shuffle(train_samples)
        for batch_id in range(0, nb_train_samples, batch_size):
            train_bar.update(batch_id)
            batch = train_samples[batch_id:batch_id+batch_size]

            # create input
            rgb_x, dep_x, y = dtp.load_data(batch, classes, DATA_LOC)
            rgb_out = np.squeeze(rgb_func(rgb_x))
            dep_out = np.squeeze(dep_func(dep_x))
            fus_in = np.concatenate((rgb_out, dep_out), axis=1)

            # train model
            loss = model.train_on_batch({'input_fus': fus_in, 'output':y})
            pred = model.predict({'input_fus': fus_in})
            acc_count, acc_on_batch = compute_accuracy(pred.get('output'), y)
            f.write(str(loss[0]) + ' --- ' + str(acc_on_batch) + '\n')
        train_bar.finish()
        elapsed_time = time.time() - start_time
        print 'Elapsed time: ' + str(elapsed_time) + 's'


        # evaluation------------------------------------------------------------------------------------
        print 'Evaluation phase'
        f.write('Evaluation phase--------------------------------------------------------------------\n')

        avg_loss = 0
        avg_accuracy = 0
        eval_bar.start()
        for batch_id in range(0, nb_eval_samples, batch_size):
            eval_bar.update(batch_id)
            batch = eval_samples[batch_id:batch_id+batch_size]

            # create input
            rgb_x, dep_x, y = dtp.load_data(batch, classes, DATA_LOC)
            rgb_out = np.squeeze(rgb_func(rgb_x))
            dep_out = np.squeeze(dep_func(dep_x))
            fus_in = np.concatenate((rgb_out, dep_out), axis=1)

            # test model
            loss = model.test_on_batch({'input_fus': fus_in, 'output':y})
            pred = model.predict({'input_fus': fus_in})
            acc_count, acc_on_batch = compute_accuracy(pred.get('output'), y)
            f.write(str(loss[0]) + ' --- ' + str(acc_on_batch) + '\n')

            # accumulate loss and accuracy
            avg_loss += loss[0]
            avg_accuracy += acc_count
        eval_bar.finish()


        # check accuracy---------------------------------------------------------------------------------
        avg_loss /= nb_eval_batches
        avg_accuracy /= nb_eval_samples
        improvement = avg_accuracy - max_accuracy
        print 'Average loss: '+str(avg_loss)+' - Average accuracy: '+str(avg_accuracy)+' - Improvement: '+str(improvement)+'\n'
        f.write('Average loss: '+str(avg_loss)+' - Average accuracy: '+str(avg_accuracy)+' - Improvement: '+str(improvement)+'\n\n')
        if max_accuracy != 0:
            improvement /= max_accuracy
        if improvement > epsilon:
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
    # load data
    classes = open(DICT, 'r').read().splitlines()
    nb_classes = len(classes)

    train_samples = dtp.sample_paths_from_list(DATA_LOC, LIST_LOC+TRAIN_LIST)
    eval_samples = dtp.sample_paths_from_list(DATA_LOC, LIST_LOC+EVAL_LIST)
    sample = [train_samples[0]]
    rgb_x, dep_x, y = dtp.load_data(sample, classes, DATA_LOC)


    # load stream models
    print 'Loading stream models...'
    rgb_model = load_model(MODEL_LOC, RGB_MODEL_NAME)
    dep_model = load_model(MODEL_LOC, DEP_MODEL_NAME)


    # get fc6
    rgb_func = theano.function([rgb_model.inputs['input_rgb'].get_input(train=False)],\
            rgb_model.nodes['fc6_rgb'].get_output(train=False))
    dep_func = theano.function([dep_model.inputs['input_dep'].get_input(train=False)],\
            dep_model.nodes['fc6_dep'].get_output(train=False))
    
    # delete models to save memory
    del rgb_model
    del dep_model

    # create fusion model
    fus_struct = create_fusion_model(nb_classes)
    fus_model = train_model(fus_struct, rgb_func, dep_func, train_samples, eval_samples, classes)
    save_model(fus_model, FUS_MODEL_NAME)

    

if __name__ == '__main__':
    main()
