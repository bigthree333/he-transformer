#import tensorflow as tf
import numpy as np
#from tensorflow.contrib.layers import Conv2D, avg_pool2d, MaxPooling2D
#from tensorflow.contrib.layers import batch_norm, l2_regularizer
#from tensorflow.contrib.framework import add_arg_scope
#from tensorflow.contrib.framework import arg_scope

import keras

from keras.models import Model, model_from_json
from keras.layers import Input, Concatenate, AveragePooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2
'''def get_variable(name, shape, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()

    if mode == 'train':
        return tf.get_variable(name, shape)
    else:
        return tf.constant(
            np.loadtxt(name + '.txt', dtype=np.float32).reshape(shape))'''


def fire_module(x,
                squeeze_depth,
                expand1_depth,
                expand3_depth,
                reuse=None,
                scope=None):
    x = Conv2D(32, (1, 1), padding="same")(x)
    x = Activation('relu')(x)

    left = Conv2D(expand1_depth, (1, 1), padding='same')(x)
    left = Activation('relu')(left)

    right = ZeroPadding2D(padding=(1, 1))(x)
    right = Conv2D(expand3_depth, (3, 3), padding='valid')(right)
    right = Activation('relu')(right)

    x = keras.layers.concatenate([left, right])
    return x


def Squeezenet(num_classes=10):
    # Simple Model ~77% accruacy
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_img)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(input=input_img, output=[x])
    return model

    # Squeeze1 from https://github.com/kaizouman/tensorsandbox/tree/master/cifar10/models/squeeze
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(
        64, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    print(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    print(x)
    x = fire_module(x, 32, 64, 64)
    print(x)
    x = fire_module(x, 32, 64, 64)
    print(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    print(x)

    x = fire_module(x, 32, 128, 128)
    print(x)

    # N x 8 x 8 x 256
    x = fire_module(x, 32, 128, 128)
    print(x)

    # Final conv to get ten classes
    # N x 8 x 8 x 10
    x = Conv2D(
        num_classes, kernel_size=(1, 1), padding='same', activation='relu')(x)
    print(x)

    # x = N x 1 x 1 x 10
    # global pooling work-around
    x = AveragePooling2D(pool_size=(8, 8))(x)
    print(x)

    y = Flatten()(x)
    print(y)

    model = Model(input=input_img, output=[y])
    return model
