from keras_marcbs.models import model_from_json
from keras_marcbs.utils.visualize_util import plot

import pdb

import keras_marcbs.caffe.convert as convert


def load_converted_model():
    location = 'pretrained/'
    json_name = 'keras_caffenet_struct.json'
    weight_name = 'keras_caffenet_weights.h5'
    #json_name = 'keras_googlenet_struct.json'
    #weight_name = 'keras_googlenet_weights.h5'

    json_str = open(location+json_name).read()
    model = model_from_json(json_str)
    model.load_weights(location+weight_name)
    pdb.set_trace()


def convert_model(save=True):
    load_path = 'keras_marcbs/caffe/models/'
    store_path = 'pretrained/'
    prototxt = 'train_val_for_keras.prototxt'

    caffemodel = 'bvlc_reference_caffenet.caffemodel'
    output_prefix = 'keras_caffenet_'

    model = convert.caffe_to_keras(load_path+prototxt, load_path+caffemodel, debug=False)

    # save model
    if save:
        json_str = model.to_json()
        open(store_path+output_prefix+'struct.json','w').write(json_str)
        model.save_weights(store_path+output_prefix+'weights.h5', overwrite=True)


if __name__ == '__main__':
    #convert_model()
    load_converted_model()
