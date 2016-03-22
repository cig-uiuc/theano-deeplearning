from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys, glob, pdb


DATA_PATH = '/media/data/washington_dataset/subset/cropped/'
DATA_LIST_TRAIN = 'data_list_train.txt'
DICT = 'dictionary.txt'
RGB_EXT = '.png'
DEP_EXT = '_depth.png'
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
  
    model.fit(x_train, y_train, nb_epoch=1, batch_size=1)

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


def colorize_depth(img):
    # scale the value from 0 to 255
    img = img.astype(float)
    img *= 255 / img.max()
    img = img.astype(np.uint8)

    # colorize depth map
    res = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    #plt.imsave('tmp.png', res)
    return res


def get_data_list():
    fid = open(DATA_LIST_TRAIN, 'r+')
    data_list = fid.readlines()
    fid.close()

    for i in range(len(data_list)):
        data_list[i] = DATA_PATH + data_list[i].rstrip()
    return data_list


def get_id_list(path):
    pwd = os.getcwd()
    os.chdir(path)
    id_list = glob.glob('*'+DEP_EXT)
    id_list = ' '.join(id_list).replace(DEP_EXT,'').split()
    id_list.sort()
    os.chdir(pwd)

    return id_list


def get_all_data_path():
    all_data_path = []
    data_list = get_data_list()
    for path in data_list:
        id_list = get_id_list(path)
        id_list.sort()
        for id in id_list:
            all_data_path.append(path+id)
    return all_data_path


def get_data(batch, categories):
    rgb_train = []
    dep_train = []
    y_train = []
    for item in batch:
        # load data
        rgb = cv2.imread(item+RGB_EXT, cv2.CV_LOAD_IMAGE_COLOR)
        dep = cv2.imread(item+DEP_EXT, cv2.CV_LOAD_IMAGE_UNCHANGED)
        
        lbl = item.split('/')[-3]
        y = [0]*len(categories)
        y[categories.index(lbl)] = 1
        
        # preprocess data
        rgb = resize_img(rgb)
        dep = colorize_depth(resize_img(dep))

        # concatenate data
        rgb_train.append(rgb)
        dep_train.append(dep)
        y_train.append(y)

    return rgb_train, dep_train, y_train


def main():
    '''
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

    # load paths 
    all_data_path = get_all_data_path()
    categories = open(DICT, 'r+').read().splitlines()

    # generate model
    model = create_model()
    RGB_model = model
    D_model = model

    # train model (by batch)
    batch_size = 10
    for batch_id in range(0, len(all_data_path), batch_size):
        batch = all_data_path[batch_id:batch_id+batch_size]
        rgb_train,dep_train,y_train = get_data(batch, categories)
        pdb.set_trace()
        
        RGB_model = train_model(RGB_model, rgb_train, y_train)
        D_model = train_model(D_model, dep_train, y_train)


if __name__ == '__main__':
    main()
