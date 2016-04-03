import keras
from keras.models import Graph
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

import pdb


def create_graph_model(nb_class):
    model = Graph()

    model.add_input(name='rgb', input_shape=(3,10,10))
    model.add_input(name='dep', input_shape=(3,10,10))

    conv_rgb_1 = Convolution2D(96,3,3,border_mode='valid')
    model.add_node(conv_rgb_1, input='rgb', name='conv_rgb_1')
    conv_dep_1 = Convolution2D(96,3,3,border_mode='valid')
    model.add_node(conv_dep_1, input='dep', name='conv_dep_1')

    model.add_node(Flatten(), input='conv_rgb_1', name='flt_rgb_1')
    model.add_node(Flatten(), input='conv_dep_1', name='flt_dep_1')

    model.add_node(Dense(100), inputs=['flt_rgb_1', 'flt_dep_1'], name='dense_1', merge_mode='concat')
    model.add_node(Dense(nb_class), input='dense_1', name='dense_2')
    model.add_output(name='output', input='dense_2')

    model.compile(loss={'output':'mse'}, optimizer='sgd')

    return model


def main():
    nb_class = 5
    model = create_graph_model(nb_class)
    plot(model, 'graph.png')

if __name__ == '__main__':
    main()

