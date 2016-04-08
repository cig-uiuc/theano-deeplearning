from keras.models import model_from_json
from keras.utils.visualize_util import plot

import pdb

import keras_marcbs.caffe.convert as convert


def load_converted_model():
    location = 'pretrained/'
    json_name = 'keras_caffenet_struct.json'
    weight_name = 'keras_caffenet_weights.h5'
    #json_name = 'keras_googlenet_struct.json'
    #weight_name = 'keras_googlenet_weights.h5'


    json_str = open(location+json_name).read()
    '''
    remove_arr = [\
            '"b_learning_rate_multiplier": null, ', \
            '"W_learning_rate_multiplier": null, ', \
            '"name": "LRN2D", ']

    for item in remove_arr:
        json_str = json_str.replace(item, '')
    '''
    
    pdb.set_trace()

    model = model_from_json(json_str)
    model.load_weights(location+weight_name)

    pdb.set_trace()


def convert_model():
    load_path = 'keras_marcbs/caffe/models/'
    store_path = 'pretrained/'
    prototxt = 'train_val_for_keras.prototxt'

    caffemodel = 'bvlc_reference_caffenet.caffemodel'
    output_prefix = 'keras_caffenet_'

    model = convert.caffe_to_keras(load_path+prototxt, load_path+caffemodel, debug=False)
    pdb.set_trace()


if __name__ == '__main__':
    convert_model()
    load_converted_model()
