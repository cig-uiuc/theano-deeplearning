from keras.models import model_from_json
import theano
import EitelModel as eitel
import data_processor as dtp
import numpy as np
import pdb
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import numpy.ma as ma

DATA_LOC = '/media/data/washington_dataset/fullset/cropped/'
LIST_LOC = './lists/'

MODEL_LOC = './models/full/'
TEST_LIST = 'test_list_full.txt'
DICT = './lists/dictionary_full.txt'

FUS_MODEL_NAME = 'fusion_model'
STRUCT_EXT = '.json'
WEIGHT_EXT = '.h5'


def load_model(loc, name):
    model = model_from_json(open(loc+name+STRUCT_EXT).read())
    model.load_weights(loc+name+WEIGHT_EXT)
    return model


def visualize_node(model, node):
    return node_func


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
                    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,\
            ncols * imshape[1] + (ncols - 1) * border),\
            dtype=np.float32)
                               
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
                                                
        mosaic[row * paddedh:row * paddedh + imshape[0],\
                col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


def make_prediction_matrix(test_samples, classes, func):
    np.random.shuffle(test_samples)
    test_samples = test_samples[:200]

    for sample in test_samples:
        rgb_x, dep_x, y = dtp.load_data([sample], classes, DATA_LOC)
        out = func(rgb_x, dep_x)
        if 'pred_mat' not in locals():
            pred_mat = out
            ground_truth = y
        else:
            pred_mat = np.concatenate((pred_mat, out), axis=0)
            ground_truth = np.concatenate((ground_truth, y), axis=0)
    return pred_mat, ground_truth


def main():
    # load data
    test_samples = dtp.sample_paths_from_list(DATA_LOC, LIST_LOC+TEST_LIST)
    sample = [test_samples[0]]
    classes = open(DICT, 'r').read().splitlines()
    nb_classes = len(classes)
    rgb_x, dep_x, y = dtp.load_data(sample, classes, DATA_LOC)

    # load model
    model = load_model(MODEL_LOC, FUS_MODEL_NAME)

    # get node functions
    node_func1 = theano.function([model.inputs['input_rgb'].get_input(train=False)],\
            model.nodes['relu1_rgb'].get_output(train=False))
    node_func2 = theano.function([model.inputs['input_rgb'].get_input(train=False)],\
            model.nodes['relu5_rgb'].get_output(train=False))

    node_func3 = theano.function([model.inputs['input_dep'].get_input(train=False)],\
            model.nodes['relu1_dep'].get_output(train=False))
    node_func4 = theano.function([model.inputs['input_dep'].get_input(train=False)],\
            model.nodes['relu5_dep'].get_output(train=False))

    node_func5 = theano.function([model.inputs['input_rgb'].get_input(train=False),\
            model.inputs['input_dep'].get_input(train=False)],\
            model.nodes['softmax_fus'].get_output(train=False) )

    out1 = np.squeeze(node_func1(rgb_x))
    out2 = np.squeeze(node_func2(rgb_x))
    out3 = np.squeeze(node_func3(dep_x))
    out4 = np.squeeze(node_func4(dep_x))
    
    # visualize intermediate nodes
    mosaic1 = make_mosaic(out1,10,10)
    fig1 = plt.figure()
    plt.imshow(mosaic1, cmap=cm.binary)
    fig1.subtitle = 'relu1_rgb'
    fig1.savefig('relu1_rgb.png')

    mosaic2 = make_mosaic(out2,16,16)
    fig2 = plt.figure()
    plt.imshow(mosaic2, cmap=cm.binary)
    fig2.subtitle = 'relu5_rgb'
    fig2.savefig('relu5_rgb')

    mosaic3 = make_mosaic(out3,10,10)
    fig3 = plt.figure()
    plt.imshow(mosaic3, cmap=cm.binary)
    fig3.subtitle = 'relu1_dep'
    fig3.savefig('relu1_dep.png')

    mosaic4 = make_mosaic(out4,16,16)
    fig4 = plt.figure()
    plt.imshow(mosaic4, cmap=cm.binary)
    fig4.subtitle = 'relu5_dep'
    fig4.savefig('relu5_dep.png')

    pred_mat, ground_truth = make_prediction_matrix(test_samples, classes, node_func5)
    fig5 = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(pred_mat)
    plt.subplot(1,2,2)
    plt.imshow(ground_truth)
    #fig5.subtitle('prediction')
    fig5.savefig('prediction.png')

    plt.show()


if __name__ == '__main__':
    main()
