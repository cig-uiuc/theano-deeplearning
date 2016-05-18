'''
Test prediction accuracy of stream and fusion models based on Eitel et al's paper.
Fusion model is still buggy.
'''
from keras.models import model_from_json
import EitelModel as eitel
import data_processor as dtp
import progressbar
import numpy as np
import pdb
import sys

DATA_LOC  = '/media/data/washington_dataset/fullset/cropped/'
LIST_LOC  = './lists/'

#MODEL_LOC = './models/small/'
#TEST_LIST = 'test_list_small.txt'
#DICT      = './lists/dictionary_small.txt'
MODEL_LOC = './models/full/'
TEST_LIST = 'test_list_full.txt'
DICT      = './lists/dictionary_full.txt'

RGB_MODEL_NAME = 'rgb_stream'
DEP_MODEL_NAME = 'dep_stream'
FUS_MODEL_NAME = 'fusion_model'
STRUCT_EXT     = '.json'
WEIGHT_EXT     = '.h5'


def load_model(loc, name):
    model = model_from_json(open(loc+name+STRUCT_EXT).read())
    model.load_weights(loc+name+WEIGHT_EXT)
    return model


def compute_accuracy(y_pred, y_true):
    N = y_pred.shape[0]
    acc_count = 0.0
    for i in range(N):
        if y_pred[i].argmax() == y_true[i].argmax():
            acc_count += 1.0
    acc_on_batch = acc_count/N
    return acc_count, acc_on_batch


def test_model(model):
    # load data
    test_samples = dtp.sample_paths_from_list(DATA_LOC, LIST_LOC+TEST_LIST)
    nb_test_samples = len(test_samples)
    classes = open(DICT, 'r').read().splitlines()
    nb_classes = len(classes)

    # test
    batch_size = 30
    avg_loss = 0
    avg_accuracy = 0
    nb_test_batches  = int(np.ceil(1.0*nb_test_samples/batch_size))
    bar = progressbar.ProgressBar(maxval=nb_test_samples, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for batch_id in range(0, nb_test_samples, batch_size):
        bar.update(batch_id)
        batch = test_samples[batch_id : batch_id+batch_size]
        rgb_x, dep_x, y = dtp.load_data(batch, classes, DATA_LOC)
        params = {'input_rgb':rgb_x, 'input_dep':dep_x, 'output':y}

        loss = model.test_on_batch(params)
        del params['output']
        pred = model.predict(params)
        acc_count, acc_on_batch = compute_accuracy(pred.get('output'), y)

        avg_loss += loss[0]
        avg_accuracy += acc_count
    bar.finish()

    # show results
    avg_loss /= nb_test_batches
    avg_accuracy /= nb_test_samples

    print 'Average loss: ' + str(avg_loss) + ' - Average accuracy: ' + str(avg_accuracy)

    return avg_loss, avg_accuracy


def main():
    # load and test models
    rgb_model = load_model(MODEL_LOC, RGB_MODEL_NAME)
    print 'Testing rgb model...'
    test_model(rgb_model)
    del rgb_model

    dep_model = load_model(MODEL_LOC, DEP_MODEL_NAME)
    print 'Testing depth model...'
    test_model(dep_model)
    del dep_model

    fus_model = load_model(MODEL_LOC, FUS_MODEL_NAME)
    print 'Testing fusion model...'
    test_model(fus_model)
    del fus_model


if __name__ == '__main__':
    main()
