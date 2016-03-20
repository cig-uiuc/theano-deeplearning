from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

import numpy as np
import matplotlib.pyplot as plt


def create_model(num_feats_in, num_feats_out, num_hidden_layers, dropouts):
    print 'Generating model architecture...'

    model = Sequential()

    # Add input layer
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

    return model


def train_model(model, e, b, v):
    print 'Training model...'

    train_input = 
    
    model.fit(train_input, train_output, nb_epoch=e, batch_size=b, verbose=v)

    return model
    

def main():
    num_feats_in = 10
    num_feats_out = 10
    num_hidden_layers = 5
    dropouts = 0.5

    model = create_model(num_feats_in, num_feats_out, num_hidden_layers, dropouts)

    model = train_model(model)


if __name__ == '__main__':
    main()
