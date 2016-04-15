import os, sys, glob, pdb
from random import shuffle

PATH = '/media/data/washington_dataset/fullset/rgbd-dataset/'
DATALIST = 'lists/data_list_full.txt'
DICT = 'lists/dictionary_full.txt'
TRAINLIST = 'lists/train_list_full.txt'
EVALLIST = 'lists/eval_list_full.txt'
TESTLIST = 'lists/test_list_full.txt'


def split():
    id_list = open(DICT, 'r').read().rsplit()
    f_train = open(TRAINLIST, 'w')
    f_eval = open(EVALLIST, 'w')
    f_test = open(TESTLIST, 'w')

    pwd = os.getcwd()
    os.chdir(PATH)

    for id in id_list:
        os.chdir(id+'/')
        objects = glob.glob('*')
        shuffle(objects)

        f_test.write(id+'/'+objects[-1]+'/\n')
        f_eval.write(id+'/'+objects[-2]+'/\n')
        for k in range(len(objects)-2):
            f_train.write(id+'/'+objects[k]+'/\n')
        os.chdir('../')

    os.chdir(pwd)
    f_train.close()
    f_eval.close()
    f_test.close()


def gen_full_list():
    f_data = open(DATALIST, 'w')
    f_dict = open(DICT, 'w')

    pwd = os.getcwd()
    os.chdir(PATH)

    id_list = glob.glob('*')
    id_list.sort()

    for id in id_list:
        f_dict.write(id+'\n')

        os.chdir(id+'/')
        objects = glob.glob('*')
        objects.sort()

        for obj in objects:
            f_data.write(id+'/'+obj+'/'+'\n')

        os.chdir('../')

    os.chdir(pwd)
    f_data.close()
    f_dict.close()


if __name__ == '__main__':
    #gen_full_list()
    split()
