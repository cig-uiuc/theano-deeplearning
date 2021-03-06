import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys, glob, pdb

RGB_EXT = '.png'
DEP_EXT = '_depth.png'
IMG_S = 227
MEAN_IMG = './pretrained/ilsvrc_2012_mean.npy'


# Data preprocessing-----------------------------------------------------------------
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
            if nb_top!=0:
                top = np.dstack([img[0,:,:]]*nb_top)
                top = top.transpose(2,0,1)
            if nb_bottom!=0:
                bottom = np.dstack([img[-1,:,:]]*nb_bottom)
                bottom = bottom.transpose(2,0,1)
        else: # depth images
            if nb_top!=0:
                top = np.tile(img[0,:], [nb_top,1])
            if nb_bottom!=0:
                bottom = np.tile(img[-1,:], [nb_bottom,1])

        # concatenate with the original image
        if nb_top==0 and nb_bottom!=0:
            res = np.concatenate((img,bottom), axis=0)
        elif nb_top!=0 and nb_bottom==0:
            res = np.concatenate((top,img), axis=0)
        else:
            res = np.concatenate((top,img,bottom), axis=0)
    elif im_w<im_h:
        new_w = int(im_w*IMG_S/im_h)
        img = cv2.resize(img, (new_w,IMG_S))

        # get number of left and right cols
        nb_left = int((IMG_S-new_w)/2)
        nb_right = IMG_S-new_w-nb_left

        # duplicate left and right cols
        if im_c==3: # color images
            if nb_left!=0:
                left = np.dstack([img[:,0,:]]*nb_left)
                left = left.transpose(0,2,1)
            if nb_right!=0:
                right = np.dstack([img[:,-1,:]]*nb_right)
                right = right.transpose(0,2,1)
        else: # depth images
            if nb_left!=0:
                left = np.tile(img[:,0], [nb_left,1]).transpose()
            if nb_right!=0:
                right = np.tile(img[:,-1], [nb_right,1]).transpose()
        
        # concatenate with the original image
        if nb_left==0 and nb_right!=0:
            res = np.concatenate((img,right), axis=1)
        elif nb_left!=0 and nb_right==0:
            res = np.concatenate((left,img), axis=1)
        else:
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
    #res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    #plt.imsave('tmp.png', res)
    return res


#Data loading---------------------------------------------------------------------
'''
def get_data_list():
    fid = open(DATA_LIST_TRAIN, 'r+')
    data_list = fid.readlines()
    fid.close()

    for i in range(len(data_list)):
        data_list[i] = DATA_PATH + data_list[i].rstrip()
    return data_list
'''

def __get_id_list__(path):
    pwd = os.getcwd()
    os.chdir(path)
    id_list = glob.glob('*'+DEP_EXT)
    id_list = ' '.join(id_list).replace(DEP_EXT,'').split()
    id_list.sort()
    os.chdir(pwd)

    return id_list

'''
def get_all_data_path():
    all_data_path = []
    data_list = get_data_list()
    for path in data_list:
        id_list = get_id_list(path)
        id_list.sort()
        for id in id_list:
            all_data_path.append(path+id)
    return all_data_path
'''

def sample_paths_from_list(data_loc, list_name):
    '''
    Return the paths to all samples (without specific extensions, e.g. _depth.png) from list_name
    '''
    paths = []
    instances = open(list_name, 'r').read().splitlines()
    for instance in instances:
        id_list = __get_id_list__(data_loc+instance)
        for id in id_list:
            paths.append(instance+id)
    return paths


def load_data(batch, categories, data_loc):
    '''
    Load data (RGB, D, and class vector) by batch
    '''
    rgb_train = []
    dep_train = []
    y_train = []

    # mean image from alexnet
    mean_img = np.load(MEAN_IMG)
    mean_img = mean_img.transpose(1,2,0)
    mean_img = cv2.resize(mean_img, (IMG_S,IMG_S))
    mean_img = mean_img.transpose(2,0,1)
    mean_img = np.float32(mean_img)

    for item in batch:
        # load data
        rgb = cv2.imread(data_loc+item+RGB_EXT, cv2.CV_LOAD_IMAGE_COLOR)

        dep = cv2.imread(data_loc+item+DEP_EXT, cv2.CV_LOAD_IMAGE_UNCHANGED)
        lbl = item.split('/')[-3]
        y = [0]*len(categories)
        y[categories.index(lbl)] = 1
        #y = categories.index(lbl)
        
        # preprocess data
        rgb = resize_img(rgb)
        dep = resize_img(colorize_depth(dep))

        '''
        id=item.split('/')[-1]
        if id=='banana_1_1_1':
            pdb.set_trace()
            cv2.imwrite('banana_1_1_1_resize.png', rgb)
            cv2.imwrite('banana_1_1_1_depth_resize.png', dep)
        '''
        
        # transpose to match model input
        rgb = rgb.transpose(2,0,1)
        dep = dep.transpose(2,0,1)

        # mean removal
        rgb = rgb - mean_img
        dep = dep - mean_img

        # concatenate data
        rgb_train.append(rgb)
        dep_train.append(dep)
        y_train.append(y)

    rgb_train = np.float32(np.array(rgb_train))
    dep_train = np.float32(np.array(dep_train))
    y_train = np.float32(np.array(y_train))

    return rgb_train, dep_train, y_train

