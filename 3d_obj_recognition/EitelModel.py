from keras.models import Graph, Sequential
from keras.layers import Dense, Dropout, Activation , Flatten, Merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
import keras.backend as K

import pdb

IMG_S = 227


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
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-5 layer
    conv5 = Convolution2D(256, 3, 3, border_mode='valid', activation='relu')
    model.add(conv5)
    model.add(MaxPooling2D(pool_size=(2,2)))
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
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-5 layer
    conv5 = Convolution2D(256, 3, 3, border_mode='valid', activation='relu', weights=trained_layers[12].get_weights())
    model.add(conv5)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # fc6 layer
    model.add(Flatten())
    model.add(Dense(4096, weights=trained_layers[16].get_weights()))
    model.add(Dropout(0.25))

    # fc7 layer
    model.add(Dense(4096, activation='softmax', weights=trained_layers[18].get_weights()))
    model.add(Dropout(0.25))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model


def create_model_merge(rgb_layers, dep_layers, nb_classes):
    rgb_stream = reuse_single_stream(rgb_layers)
    dep_stream = reuse_single_stream(dep_layers)

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
    #model.compile(loss=fus_loss, optimizer='sgd')

    return model


def train_model(model, x_train, y_train, b_size):
    model.fit(x_train, y_train, nb_epoch=1, batch_size=b_size)
    return model


def stream_loss(x):
    return x

def fus_loss(y_true, y_pred):
    # reuse backend's mean squared errors
    return K.mean(K.square(y_true - y_pred), axis=-1)
