import os, sys, glob, pdb

PATH = '/media/data/washington_dataset/fullset/rgbd-dataset/'
DATALIST = 'lists/data_list_full.txt'
DICT = 'lists/dictionary_full.txt'


def main():
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
            f_data.write(id+'/'+obj+'\n')

        os.chdir('../')

    os.chdir(pwd)
    f_data.close()
    f_dict.close()


if __name__ == '__main__':
    main()
