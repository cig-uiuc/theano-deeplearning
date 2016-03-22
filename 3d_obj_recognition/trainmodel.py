from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb

IMG_S = 227


def create_model():
    print 'Generating model architecture...'

    model = Sequential()

    # input layer???
    #model.add(Dense())
    
    # conv-1 layer
    model.add(Convolution2D(96, 3, 3, border_mode='valid', input_shape=(3,IMG_S,IMG_S)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-2 layer
    model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # conv-3 layer
    model.add(Convolution2D(384, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv-4 layer
    model.add(Convolution2D(384, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # conv-5 layer
    model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # fc6 layer
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # fc7 layer
    model.add(Dense(4096))
    model.add(Activation('softmax'))

    # compile model
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    

    '''
    model.add(Dense(output_dim=num_feats_out, input_dim=num_feats_in, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(dropouts))

    # Add hidden layers
    for i in range(num_hidden_layers):
        model.add(Dense(output_dim=num_feats_out))
        model.add(Activation('relu'))
        model.add(Dropout(dropouts))

    # Add output layers
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))

    # Compile model
    model.compile(loss='mse', optimizer='sgd')
    '''

    return model


def train_model(model):
    print 'Training model...'

    
    #model.fit(train_input, train_output, nb_epoch=e, batch_size=b, verbose=v)

    return model


def resize_img(img):
    if len(img.shape)==3:
        im_h,im_w,im_c = img.shape
    else:
        im_h,im_w = img.shape
        im_c = 1

    # resize image
    if im_w>im_h:
        new_h = int(im_h*IMG_S/im_w)
        img = cv2.resize(img, (IMG_S,new_h))

        # get number of top and bottom rows
        nb_top = int((IMG_S-new_h)/2)
        nb_bottom = IMG_S-new_h-nb_top
        
        # duplicate top and bottom rows
        if im_c==3: # color images
            top = np.dstack([img[0,:,:]]*nb_top)
            bottom = np.dstack([img[-1,:,:]]*nb_bottom)
            top = top.transpose(2,0,1)
            bottom = bottom.transpose(2,0,1)
        else: # depth images
            top = np.tile(img[0,:], [nb_top,1])
            bottom = np.tile(img[-1,:], [nb_bottom,1])

        # concatenate with the original image
        res = np.concatenate((top,img,bottom), axis=0)
    elif im_w<im_h:
        new_w = int(im_w*IMG_S/im_h)
        img = cv2.resize(img, (new_w,IMG_S))

        # get number of left and right cols
        nb_left = int((IMG_S-new_w)/2)
        nb_right = IMG_S-new_w-nb_left

        # duplicate left and right cols
        if im_c==3: # color images
            left = np.dstack([img[:,0,:]]*nb_left)
            right = np.dstack([img[:,-1,:]]*nb_right)
            left = left.transpose(0,2,1)
            right = right.transpose(0,2,1)
        else: # depth images
            left = np.tile(img[:,0], [nb_left,1]).transpose()
            right = np.tile(img[:,-1], [nb_right,1]).transpose()

        res = np.concatenate((left,img,right), axis=1)
    else:
        res = cv2.resize(img, (IMG_S,IMG_S))

    return res


def coloize_depth(img):
    # scale the value from 0 to 255
    img = img.astype(float)
    img *= 255 / img.max()
    img = img.astype(np.uint8)

    # colorize depth map
    res = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    #plt.imsave('tmp.png', res)
    return res


def main():
    # dummy inputs (1 instance)
    tmp_path = '/media/data/washington_dataset/subset/cropped/banana/banana_1/'
    rgb_name = tmp_path+'banana_1_1_1.png'
    dep_name = tmp_path+'banana_1_1_1_depth.png'
    rgb = cv2.imread(rgb_name, cv2.CV_LOAD_IMAGE_COLOR)
    dep = cv2.imread(dep_name, cv2.CV_LOAD_IMAGE_UNCHANGED)
    
    # preprocess data
    rgb = resize_img(rgb)
    dep = resize_img(dep)
    dep = colorize_depth(dep)
    
    



    '''
    # generate model
    archi = create_model()
    RGB_model = train_model(archi)
    D_model = train_model(archi)
    '''


if __name__ == '__main__':
    main()
