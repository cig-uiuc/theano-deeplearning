import os, sys, glob, pdb

import numpy as np
import matplotlib.pyplot as plt
import cv2

DATA_LIST  = 'data_list.txt'
INPUT_DIR  = '/media/data/washington_dataset/subset/rgbd-dataset/'
OUTPUT_DIR = '/media/data/washington_dataset/subset/cropped/'

RGB_EXT  = '.png'
DEP_EXT  = '_depth.png'
MASK_EXT = '_mask.png'
LOC_EXT  = '_loc.txt'

OBJ_W = 75
OBJ_H = 75


def get_data_list():
    fid = open(DATA_LIST, 'r+')
    data_list = fid.readlines()
    fid.close()

    for i in xrange(0,len(data_list)):
        data_list[i] = INPUT_DIR + data_list[i].rstrip()
    return data_list


def get_id_list(path):
    os.chdir(path)
    id_list = glob.glob('*'+LOC_EXT)
    id_list = ' '.join(id_list).replace(LOC_EXT,'').split()
    return id_list


def process(path, id):
    # make directories to store output
    out_path = path.replace(INPUT_DIR, OUTPUT_DIR)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # generate names
    rgb_name = path + id + RGB_EXT
    dep_name = path + id + DEP_EXT
    mask_name = path + id + MASK_EXT
    loc_name = path + id + LOC_EXT
    
    # get data
    rgb = cv2.imread(rgb_name, cv2.CV_LOAD_IMAGE_COLOR)
    dep = cv2.imread(dep_name, cv2.CV_LOAD_IMAGE_UNCHANGED)
    mask = cv2.imread(mask_name, cv2.CV_LOAD_IMAGE_UNCHANGED)
    mask = np.divide(mask, 255)

    f = open(loc_name, 'r+')
    line = f.readline()
    f.close()
    line = line.rstrip()
    x0 = int(line.split(',')[0])
    y0 = int(line.split(',')[1])

    # apply mask
    rgb[mask==0] = 0
    dep[mask==0] = 0

    # crop
    rgb_crop = rgb[y0:y0+OBJ_H, x0:x0+OBJ_W]
    dep_crop = dep[y0:y0+OBJ_H, x0:x0+OBJ_W]
    
    # write to file
    cv2.imwrite(out_path+id+RGB_EXT, rgb_crop)
    cv2.imwrite(out_path+id+DEP_EXT, dep_crop)
    #pdb.set_trace()


if __name__ == '__main__':
    data_list = get_data_list()
    for path in data_list:
        print path
        id_list = get_id_list(path)
        for id in id_list:
            process(path, id)

