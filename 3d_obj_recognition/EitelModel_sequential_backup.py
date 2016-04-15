from keras.models import Graph, Sequential
from keras.layers import Dense, Dropout, Activation , Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization#, LRN2D
from keras.optimizers import SGD, Adagrad
from keras.utils.visualize_util import plot

import pdb

IMG_S = 227

def weights_from_graph(pretrained):
    pre_conv1 = pretrained.nodes.get('conv1').get_weights()
    pre_conv2 = pretrained.nodes.get('conv2').get_weights()
    pre_conv3 = pretrained.nodes.get('conv3').get_weights()
    pre_conv4 = pretrained.nodes.get('conv4').get_weights()
    pre_conv5 = pretrained.nodes.get('conv5').get_weights()
    pre_fc6 = pretrained.nodes.get('fc6').get_weights()
    pre_fc7 = pretrained.nodes.get('fc7').get_weights()
    return (pre_conv1,pre_conv2,pre_conv3,pre_conv4,pre_conv5,pre_fc6,pre_fc7)


def weights_from_sequential(model):
    pre_conv1 = model.layers[0].get_weights()
    pre_conv2 = model.layers[5].get_weights()
    pre_conv3 = model.layers[10].get_weights()
    pre_conv4 = model.layers[13].get_weights()
    pre_conv5 = model.layers[16].get_weights()
    pre_fc6 = model.layers[20].get_weights()
    pre_fc7 = model.layers[23].get_weights()
    return (pre_conv1,pre_conv2,pre_conv3,pre_conv4,pre_conv5,pre_fc6,pre_fc7)


def create_single_stream(nb_classes, inmodel, mode):
    # load pretrained's nodes
    if mode==0: # use pretrained weights from AlexNet
        (pre_conv1,pre_conv2,pre_conv3,pre_conv4,pre_conv5,pre_fc6,pre_fc7) = weights_from_graph(inmodel)
        totrain = True
    elif mode==1: # use our trained weights from sequential model
        (pre_conv1,pre_conv2,pre_conv3,pre_conv4,pre_conv5,pre_fc6,pre_fc7) = weights_from_sequential(inmodel)
        totrain = False

    # create model
    model = Sequential()

    # conv-1 layer
    conv1 = Convolution2D(96, 11, 11, border_mode='valid', subsample=(4,4), input_shape=(3,IMG_S,IMG_S), name='conv1', weights=pre_conv1, trainable=totrain)
    relu1 = Activation('relu', name='relu1')
    norm1 = BatchNormalization(name='norm1')
    #norm1 = LRN2D(name='norm1')
    pool1 = MaxPooling2D(pool_size=(2,2), name='pool1')

    model.add(conv1)
    model.add(relu1)
    model.add(norm1)
    model.add(pool1)

    # conv-2 layer
    conv2_zeropadding = ZeroPadding2D(padding=(2,2), name='conv2_zeropadding')
    conv2 = Convolution2D(256, 5, 5, border_mode='valid', name='conv2', weights=pre_conv2, trainable=totrain)
    relu2 = Activation('relu', name='relu2')
    norm2 = BatchNormalization(name='norm2')
    #norm2 = LRN2D(name='norm2')
    pool2 = MaxPooling2D(pool_size=(2,2), name='pool2')

    model.add(conv2_zeropadding)
    model.add(conv2)
    model.add(relu2)
    model.add(norm2)
    model.add(pool2)


    # conv-3 layer
    conv3_zeropadding = ZeroPadding2D(padding=(1,1), name='conv3_zeropadding')
    conv3 = Convolution2D(384, 3, 3, border_mode='valid', name='conv3', weights=pre_conv3, trainable=totrain)
    relu3 = Activation('relu', name='relu3')

    model.add(conv3_zeropadding)
    model.add(conv3)
    model.add(relu3)


    # conv-4 layer
    conv4_zeropadding = ZeroPadding2D(padding=(1,1), name='conv4_zeropadding')
    conv4 = Convolution2D(384, 3, 3, border_mode='valid', name='conv4', weights=pre_conv4, trainable=totrain)
    relu4 = Activation('relu', name='relu4')

    model.add(conv4_zeropadding)
    model.add(conv4)
    model.add(relu4)


    # conv-5 layer
    conv5_zeropadding = ZeroPadding2D(padding=(1,1), name='conv5_zeropadding')
    conv5 = Convolution2D(256, 3, 3, border_mode='valid', name='conv5', weights=pre_conv5, trainable=totrain)
    relu5 = Activation('relu', name='relu5')
    pool5 = MaxPooling2D(pool_size=(2,2), name='pool5')

    model.add(conv5_zeropadding)
    model.add(conv5)
    model.add(relu5)
    model.add(pool5)
   

    # fc6 layer
    fc6_flatten = Flatten(name='fc6_flatten')
    fc6 = Dense(4096, name='fc6', weights=pre_fc6, trainable=totrain)
    relu6 = Activation('relu', name='relu6')
    drop6 = Dropout(0.5, name='drop6')

    model.add(fc6_flatten)
    model.add(fc6)
    model.add(relu6)
    model.add(drop6)


    # fc7 layer
    fc7 = Dense(4096, name='fc7', weights=pre_fc7, trainable=totrain)
    relu7 = Activation('relu', name='relu7')
    drop7 = Dropout(0.5, name='drop7')

    model.add(fc7)
    model.add(relu7)
    model.add(drop7)

    if mode==0: # from pretrained AlexNet
        # classifier layer
        fc8 = Dense(nb_classes, name='fc8')
        output_loss = Activation('softmax', name='softmax')

        model.add(fc8)
        model.add(output_loss)


        # compile model
        sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        #model.compile(loss='categorical_crossentropy', optimizer='adagrad')

    return model


def create_model_merge(in_rgb, in_dep, nb_classes):
    rgb_stream = create_single_stream(nb_classes, in_rgb, mode=1)
    dep_stream = create_single_stream(nb_classes, in_dep, mode=1)

    model = Sequential()

    # fc1-fus layer
    fc1_fus = Merge([rgb_stream, dep_stream], mode='concat')
    model.add(fc1_fus)
    model.add(Dense(4096))
    model.add(Dropout(0.5))
    
    # classifier layer
    model.add(Dense(nb_classes, activation='softmax'))

    # compile model
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    #model.compile(loss='categorical_crossentropy', optimizer='adagrad')

    return model

'''
def stream_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def fus_loss(y_true, y_pred):
    # dummy code: reuse backend's categorical crossentropy
    return K.categorical_crossentropy(y_true, y_pred)
'''
