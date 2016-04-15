from keras.models import Graph, Sequential
from keras.layers import Dense, Dropout, Activation , Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization#, LRN2D
from keras.optimizers import SGD, Adagrad
from keras.utils.visualize_util import plot

import pdb

IMG_S = 227

def weights_from_graph(pretrained, tag):
    pre_conv1 = pretrained.nodes.get('conv1'+tag).get_weights()
    pre_conv2 = pretrained.nodes.get('conv2'+tag).get_weights()
    pre_conv3 = pretrained.nodes.get('conv3'+tag).get_weights()
    pre_conv4 = pretrained.nodes.get('conv4'+tag).get_weights()
    pre_conv5 = pretrained.nodes.get('conv5'+tag).get_weights()
    pre_fc6 = pretrained.nodes.get('fc6'+tag).get_weights()
    pre_fc7 = pretrained.nodes.get('fc7'+tag).get_weights()
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


def create_single_stream(nb_classes, inmodel, tag):
    # load pretrained's nodes
    #if mode==0: # use pretrained weights from AlexNet
    #    (pre_conv1,pre_conv2,pre_conv3,pre_conv4,pre_conv5,pre_fc6,pre_fc7) = weights_from_graph(inmodel, '')
    #    totrain = True
    #elif mode==1: # use our trained weights from sequential model
    #    (pre_conv1,pre_conv2,pre_conv3,pre_conv4,pre_conv5,pre_fc6,pre_fc7) = weights_from_sequential(inmodel, tag)
    #    totrain = False
    (pre_conv1,pre_conv2,pre_conv3,pre_conv4,pre_conv5,pre_fc6,pre_fc7) = weights_from_graph(inmodel, '')
    totrain=True

    # create model
    model = Graph()
    model.add_input(name='input'+tag, input_shape=(3,IMG_S,IMG_S))

    # conv-1 layer
    conv1 = Convolution2D(96, 11, 11, border_mode='valid', subsample=(4,4), weights=pre_conv1, trainable=totrain)
    relu1 = Activation('relu')
    norm1 = BatchNormalization()
    pool1 = MaxPooling2D(pool_size=(2,2))

    model.add_node(conv1, name='conv1'+tag, input='input'+tag)
    model.add_node(relu1, name='relu1'+tag, input='conv1'+tag)
    model.add_node(norm1, name='norm1'+tag, input='relu1'+tag)
    model.add_node(pool1, name='pool1'+tag, input='norm1'+tag)
    #model.add(conv1)
    #model.add(relu1)
    #model.add(norm1)
    #model.add(pool1)

    # conv-2 layer
    conv2_zeropadding = ZeroPadding2D(padding=(2,2))
    conv2 = Convolution2D(256, 5, 5, border_mode='valid', weights=pre_conv2, trainable=totrain)
    relu2 = Activation('relu')
    norm2 = BatchNormalization()
    pool2 = MaxPooling2D(pool_size=(2,2))

    model.add_node(conv2_zeropadding, name='conv2_zeropadding'+tag, input='pool1'+tag)
    model.add_node(conv2, name='conv2'+tag, input='conv2_zeropadding'+tag)
    model.add_node(relu2, name='relu2'+tag, input='conv2'+tag)
    model.add_node(norm2, name='norm2'+tag, input='relu2'+tag)
    model.add_node(pool2, name='pool2'+tag, input='norm2'+tag)
    #model.add(conv2_zeropadding)
    #model.add(conv2)
    #model.add(relu2)
    #model.add(norm2)
    #model.add(pool2)


    # conv-3 layer
    conv3_zeropadding = ZeroPadding2D(padding=(1,1))
    conv3 = Convolution2D(384, 3, 3, border_mode='valid', weights=pre_conv3, trainable=totrain)
    relu3 = Activation('relu')

    model.add_node(conv3_zeropadding, name='conv3_zeropadding'+tag, input='pool2'+tag)
    model.add_node(conv3, name='conv3'+tag, input='conv3_zeropadding'+tag) 
    model.add_node(relu3, name='relu3'+tag, input='conv3'+tag) 
    #model.add(conv3_zeropadding)
    #model.add(conv3)
    #model.add(relu3)


    # conv-4 layer
    conv4_zeropadding = ZeroPadding2D(padding=(1,1))
    conv4 = Convolution2D(384, 3, 3, border_mode='valid', weights=pre_conv4, trainable=totrain)
    relu4 = Activation('relu')

    model.add_node(conv4_zeropadding, name='conv4_zeropadding'+tag, input='relu3'+tag) 
    model.add_node(conv4, name='conv4'+tag, input='conv4_zeropadding'+tag) 
    model.add_node(relu4, name='relu4'+tag, input='conv4'+tag) 
    #model.add(conv4_zeropadding)
    #model.add(conv4)
    #model.add(relu4)


    # conv-5 layer
    conv5_zeropadding = ZeroPadding2D(padding=(1,1))
    conv5 = Convolution2D(256, 3, 3, border_mode='valid', weights=pre_conv5, trainable=totrain)
    relu5 = Activation('relu')
    pool5 = MaxPooling2D(pool_size=(2,2))

    model.add_node(conv5_zeropadding, name='conv5_zeropadding'+tag, input='relu4'+tag) 
    model.add_node(conv5, name='conv5'+tag, input='conv5_zeropadding'+tag) 
    model.add_node(relu5, name='relu5'+tag, input='conv5'+tag) 
    model.add_node(pool5, name='pool5'+tag, input='relu5'+tag) 
    #model.add(conv5_zeropadding)
    #model.add(conv5)
    #model.add(relu5)
    #model.add(pool5)
   

    # fc6 layer
    fc6_flatten = Flatten()
    fc6 = Dense(4096, weights=pre_fc6, trainable=totrain)
    relu6 = Activation('relu')
    drop6 = Dropout(0.5)

    model.add_node(fc6_flatten, name='fc6_flatten'+tag, input='pool5'+tag) 
    model.add_node(fc6, name='fc6'+tag, input='fc6_flatten'+tag) 
    model.add_node(relu6, name='relu6'+tag, input='fc6'+tag) 
    model.add_node(drop6, name='drop6'+tag, input='relu6'+tag) 
    #model.add(fc6_flatten)
    #model.add(fc6)
    #model.add(relu6)
    #model.add(drop6)


    # fc7 layer
    fc7 = Dense(4096, weights=pre_fc7, trainable=totrain)
    relu7 = Activation('relu')
    drop7 = Dropout(0.5)

    model.add_node(fc7, name='fc7'+tag, input='drop6'+tag) 
    model.add_node(relu7, name='relu7'+tag, input='fc7'+tag) 
    model.add_node(drop7, name='drop7'+tag, input='relu7'+tag) 
    #model.add(fc7)
    #model.add(relu7)
    #model.add(drop7)

    #if mode==0: # from pretrained AlexNet
    # classifier layer
    fc8 = Dense(nb_classes)
    output_loss = Activation('softmax', name='output_loss')

    model.add_node(fc8, name='fc8'+tag, input='drop7'+tag) 
    model.add_node(output_loss, name='output_loss'+tag, input='fc8'+tag) 
    model.add_output(name='output', input='output_loss'+tag)
    #model.add(fc8)
    #model.add(output_loss)

    # compile model
    if tag=='_rgb':
        learning_rate = 0.01
    elif tag=='_dep':
        learning_rate = 0.003
    else:
        learning_rate = 0.01
    sgd = SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss={'output':'categorical_crossentropy'}, optimizer=sgd)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd)
    #model.compile(loss='categorical_crossentropy', optimizer='adagrad')

    return model


def construct_branch(model, branch, tag):
    # load data from branch
    conv1 = branch.nodes.get('conv1'+tag)
    relu1 = branch.nodes.get('relu1'+tag)
    norm1 = branch.nodes.get('norm1'+tag)
    pool1 = branch.nodes.get('pool1'+tag)
    conv2_zeropadding = branch.nodes.get('conv2_zeropadding'+tag)
    conv2 = branch.nodes.get('conv2'+tag)
    relu2 = branch.nodes.get('relu2'+tag)
    norm2 = branch.nodes.get('norm2'+tag)
    pool2 = branch.nodes.get('pool2'+tag)
    conv3_zeropadding = branch.nodes.get('conv3_zeropadding'+tag)
    conv3 = branch.nodes.get('conv3'+tag)
    relu3 = branch.nodes.get('relu3'+tag)
    conv4_zeropadding = branch.nodes.get('conv4_zeropadding'+tag)
    conv4 = branch.nodes.get('conv4'+tag)
    relu4 = branch.nodes.get('relu4'+tag)
    conv5_zeropadding = branch.nodes.get('conv5_zeropadding'+tag)
    conv5 = branch.nodes.get('conv5'+tag)
    relu5 = branch.nodes.get('relu5'+tag)
    pool5 = branch.nodes.get('pool5'+tag)
    fc6_flatten = branch.nodes.get('fc6_flatten'+tag)
    fc6 = branch.nodes.get('fc6'+tag)
    relu6 = branch.nodes.get('relu6'+tag)
    drop6 = branch.nodes.get('drop6'+tag)
    fc7 = branch.nodes.get('fc7'+tag)
    relu7 = branch.nodes.get('relu7'+tag)
    drop7 = branch.nodes.get('drop7'+tag)

    # don't allow training on branch
    conv1.trainable = False
    conv2.trainable = False
    conv3.trainable = False
    conv4.trainable = False
    conv5.trainable = False

    norm1.trainable = False
    norm2.trainable - False

    # add nodes
    model.add_input(name='input'+tag, input_shape=(3,IMG_S,IMG_S))
    model.add_node(conv1, name=conv1._name, input='input'+tag)
    model.add_node(relu1, name='relu1'+tag, input='conv1'+tag)
    model.add_node(norm1, name='norm1'+tag, input='relu1'+tag)
    model.add_node(pool1, name='pool1'+tag, input='norm1'+tag)

    model.add_node(conv2_zeropadding, name='conv2_zeropadding'+tag, input='pool1'+tag)
    model.add_node(conv2, name='conv2'+tag, input='conv2_zeropadding'+tag)
    model.add_node(relu2, name='relu2'+tag, input='conv2'+tag)
    model.add_node(norm2, name='norm2'+tag, input='relu2'+tag)
    model.add_node(pool2, name='pool2'+tag, input='norm2'+tag)
    
    model.add_node(conv3_zeropadding, name='conv3_zeropadding'+tag, input='pool2'+tag)
    model.add_node(conv3, name='conv3'+tag, input='conv3_zeropadding'+tag) 
    model.add_node(relu3, name='relu3'+tag, input='conv3'+tag)

    model.add_node(conv4_zeropadding, name='conv4_zeropadding'+tag, input='relu3'+tag) 
    model.add_node(conv4, name='conv4'+tag, input='conv4_zeropadding'+tag) 
    model.add_node(relu4, name='relu4'+tag, input='conv4'+tag) 

    model.add_node(conv5_zeropadding, name='conv5_zeropadding'+tag, input='relu4'+tag) 
    model.add_node(conv5, name='conv5'+tag, input='conv5_zeropadding'+tag) 
    model.add_node(relu5, name='relu5'+tag, input='conv5'+tag) 
    model.add_node(pool5, name='pool5'+tag, input='relu5'+tag) 

    model.add_node(fc6_flatten, name='fc6_flatten'+tag, input='pool5'+tag) 
    model.add_node(fc6, name='fc6'+tag, input='fc6_flatten'+tag) 
    model.add_node(relu6, name='relu6'+tag, input='fc6'+tag) 
    model.add_node(drop6, name='drop6'+tag, input='relu6'+tag) 

    model.add_node(fc7, name='fc7'+tag, input='drop6'+tag) 
    model.add_node(relu7, name='relu7'+tag, input='fc7'+tag) 
    model.add_node(drop7, name='drop7'+tag, input='relu7'+tag) 

    return model

def create_model_merge(in_rgb, in_dep, nb_classes):
    #rgb_stream = create_single_stream(nb_classes, in_rgb, mode=1, tag='_rgb')
    #dep_stream = create_single_stream(nb_classes, in_dep, mode=1, tag='_dep')

    model = Graph()
    model = construct_branch(model, in_rgb, '_rgb')
    model = construct_branch(model, in_dep, '_dep')

    # fc1-fus layer
    #fc1_fus = Merge([rgb_stream, dep_stream], mode='concat')
    #dense_fus = Dense(4096)
    #drop_fus = Dropout(0.5)

    model.add_node(Dense(4096), name='fc1_fus', inputs=['drop7_rgb', 'drop7_dep'], merge_mode='concat')
    #model.add_node(Dropout(0.5), name='drop_fus', input='fc1_fus')

    #model.add(fc1_fus)
    #model.add(dense_fus)
    #model.add(drop_fus)
    
    # classifier layer
    #model.add(Dense(nb_classes, activation='softmax'))
    model.add_node(Dense(nb_classes, activation='softmax'), name='softmax_fus', input='fc1_fus')
    model.add_output(name='output', input='softmax_fus')

    # compile model
    sgd = SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss={'output':'categorical_crossentropy'}, optimizer=sgd)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd)
    #model.compile(loss='categorical_crossentropy', optimizer='adagrad')

    return model

'''
def stream_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def fus_loss(y_true, y_pred):
    # dummy code: reuse backend's categorical crossentropy
    return K.categorical_crossentropy(y_true, y_pred)
'''
