from keras.models import Graph, Sequential
from keras.layers import Dense, Dropout, Activation , Flatten, Merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

import pdb

IMG_S = 227

'''
def create_model_graph(nb_classes):
    model = Graph()

    # input layers
    model.add_input(name='rgb_input', input_shape=(3,IMG_S,IMG_S))
    model.add_input(name='dep_input', input_shape=(3,IMG_S,IMG_S))

    # conv-1 layers
    rgb_conv1 = Convolution2D(96, 3, 3, border_mode='valid', activation='relu')
    model.add_node(rgb_conv1, input='rgb_input', name='rgb_conv1')
    model.add_node(MaxPooling2D(pool_size=(2,2)), input='rgb_conv1', name='rgb_mp1')
    model.add_node(Dropout(0.25), input='rgb_mp1', name='rgb_conv1_out')

    dep_conv1 = Convolution2D(96, 3, 3, border_mode='valid', activation='relu')
    model.add_node(dep_conv1, input='dep_input', name='dep_conv1')
    model.add_node(MaxPooling2D(pool_size=(2,2)), input='dep_conv1', name='dep_mp1')
    model.add_node(Dropout(0.25), input='dep_mp1', name='dep_conv1_out')

    # conv-2 layers
    rgb_conv2 = Convolution2D(256, 3, 3, border_mode='valid', activation='relu')
    model.add_node(rgb_conv2, input='rgb_conv1_out', name='rgb_conv2')
    model.add_node(MaxPooling2D(pool_size=(2,2)), input='rgb_conv2', name='rgb_mp2')
    model.add_node(Dropout(0.25), input='rgb_mp2', name='rgb_conv2_out')

    dep_conv2 = Convolution2D(256, 3, 3, border_mode='valid', activation='relu')
    model.add_node(dep_conv2, input='dep_conv1_out', name='dep_conv2')
    model.add_node(MaxPooling2D(pool_size=(2,2)), input='dep_conv2', name='dep_mp2')
    model.add_node(Dropout(0.25), input='dep_mp2', name='dep_conv2_out')

    # conv-3 layers
    rgb_conv3 = Convolution2D(384, 3, 3, border_mode='valid', activation='relu')
    model.add_node(rgb_conv3, input='rgb_conv2_out', name='rgb_conv3')
    model.add_node(MaxPooling2D(pool_size=(2,2)), input='rgb_conv3', name='rgb_mp3')
    model.add_node(Dropout(0.25), input='rgb_mp3', name='rgb_conv3_out')

    dep_conv3 = Convolution2D(384, 3, 3, border_mode='valid', activation='relu')
    model.add_node(dep_conv3, input='dep_conv2_out', name='dep_conv3')
    model.add_node(MaxPooling2D(pool_size=(2,2)), input='dep_conv3', name='dep_mp3')
    model.add_node(Dropout(0.25), input='dep_mp3', name='dep_conv3_out')

    # conv-4 layers
    rgb_conv4 = Convolution2D(384, 3, 3, border_mode='valid', activation='relu')
    model.add_node(rgb_conv4, input='rgb_conv3_out', name='rgb_conv4')
    model.add_node(Dropout(0.25), input='rgb_conv4', name='rgb_conv4_out')

    dep_conv4 = Convolution2D(384, 3, 3, border_mode='valid', activation='relu')
    model.add_node(dep_conv4, input='dep_conv3_out', name='dep_conv4')
    model.add_node(Dropout(0.25), input='dep_conv3', name='dep_conv4_out')
    
    # conv-5 layers
    rgb_conv5 = Convolution2D(256, 3, 3, border_mode='valid', activation='relu')
    model.add_node(rgb_conv5, input='rgb_conv4_out', name='rgb_conv5')
    model.add_node(Dropout(0.25), input='rgb_conv5', name='rgb_conv5_out')

    dep_conv5 = Convolution2D(256, 3, 3, border_mode='valid', activation='relu')
    model.add_node(dep_conv5, input='dep_conv4_out', name='dep_conv5')
    model.add_node(Dropout(0.25), input='dep_conv5', name='dep_conv5_out')

    # fc6 layers
    model.add_node(Flatten(), input='rgb_conv5_out', name='rgb_flatten')
    model.add_node(Dense(4096), input='rgb_flatten', name='rgb_fc6')
    model.add_node(Dropout(0.25), input='rgb_fc6', name='rgb_fc6_out')

    model.add_node(Flatten(), input='dep_conv5_out', name='dep_flatten')
    model.add_node(Dense(4096), input='dep_flatten', name='dep_fc6')
    model.add_node(Dropout(0.25), input='dep_fc6', name='dep_fc6_out')

    # fc7 layers
    model.add_node(Dense(4096), input='rgb_fc6_out', name='rgb_fc7')
    model.add_node(Dropout(0.25), input='rgb_fc7', name='rgb_fc7_out')

    model.add_node(Dense(4096), input='dep_fc6_out', name='dep_fc7')
    model.add_node(Dropout(0.25), input='dep_fc7', name='dep_fc7_out')


    # fc1-fus layer
    model.add_node(Dense(4096), inputs=['rgb_fc7_out', 'dep_fc7_out'], name='fc1_fus', merge_mode='concat')

    # classifier layer
    model.add_node(Dense(nb_classes, activation='softmax'), input='fc1_fus', name='class')
    model.add_output(input='class', name='output')

    # compile model
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss={'output':'categorical_crossentropy'}, optimizer='sgd')

    return model
'''

def create_single_stream(nb_classes):
    model = Sequential()

    # conv-1 layer
    conv1 = Convolution2D(96, 3, 3, border_mode='valid', activation='relu', input_shape=(3,IMG_S,IMG_S))
    model.add(conv1)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-2 layer
    conv2 = Convolution2D(256, 3, 3, border_mode='valid', activation='relu')
    model.add(conv2)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-3 layer
    conv3 = Convolution2D(384, 3, 3, border_mode='valid', activation='relu')
    model.add(conv3)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-4 layer
    conv4 = Convolution2D(384, 3, 3, border_mode='valid', activation='relu')
    model.add(conv4)
    model.add(Dropout(0.25))

    # conv-5 layer
    conv5 = Convolution2D(256, 3, 3, border_mode='valid', activation='relu')
    model.add(conv5)
    model.add(Dropout(0.25))

    # fc6 layer
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dropout(0.25))

    # fc7 layer
    model.add(Dense(4096))
    model.add(Dropout(0.25))

    # classifier layer
    model.add(Dense(nb_classes, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model


def reuse_single_stream(trained_layers):
    model = Sequential()

    # conv-1 layer
    conv1 = Convolution2D(96, 3, 3, border_mode='valid', activation='relu', input_shape=(3,IMG_S,IMG_S), weights=trained_layers[0].get_weights())
    model.add(conv1)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-2 layer
    conv2 = Convolution2D(256, 3, 3, border_mode='valid', activation='relu', weights=trained_layers[3].get_weights())
    model.add(conv2)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-3 layer
    conv3 = Convolution2D(384, 3, 3, border_mode='valid', activation='relu', weights=trained_layers[6].get_weights())
    model.add(conv3)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-4 layer
    conv4 = Convolution2D(384, 3, 3, border_mode='valid', activation='relu', weights=trained_layers[9].get_weights())
    model.add(conv4)
    model.add(Dropout(0.25))

    # conv-5 layer
    conv5 = Convolution2D(256, 3, 3, border_mode='valid', activation='relu', weights=trained_layers[11].get_weights())
    model.add(conv5)
    model.add(Dropout(0.25))

    # fc6 layer
    model.add(Flatten())
    model.add(Dense(4096, weights=trained_layers[14].get_weights()))
    model.add(Dropout(0.25))

    # fc7 layer
    model.add(Dense(4096, activation='softmax', weights=trained_layers[16].get_weights()))
    model.add(Dropout(0.25))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model


def create_model_merge(trained_rgb_stream, trained_dep_stream, nb_classes):
    rgb_stream = reuse_single_stream(trained_rgb_stream.layers)
    dep_stream = reuse_single_stream(trained_dep_stream.layers)

    model = Sequential()

    # fc1-fus layer
    fc1_fus = Merge([rgb_stream, dep_stream], mode='concat')
    model.add(fc1_fus)
    model.add(Dense(4096))
    model.add(Dropout(0.5))
    
    # classifier layer
    model.add(Dense(nb_classes, activation='softmax'))
    

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    return model


def train_model(model, x_train, y_train, b_size):
    model.fit(x_train, y_train, nb_epoch=1, batch_size=b_size)
    return model


def stream_activation():
    return

def fus_activation():
    return
