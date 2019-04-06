# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
# CPU needs NHWC format for MaxPool / FusedBatchNorm
keras.backend.set_image_data_format('channels_last')

from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import numpy as np
import squeezenet
#import tensorflow as tf
import time
from datetime import datetime
import os
import argparse
import sys

FLAGS = None


def main():
    # Import data
    num_classes = 10
    epochs = 1
    NUM_TRAIN_SAMPLES = 50000
    IMAGE_HEIGHT = IMAGE_WIDTH = 32
    CHANNELS = 3

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, IMAGE_HEIGHT,
                                  IMAGE_WIDTH)
        x_test = x_test.reshape(x_test.shape[0], 3, IMAGE_WIDTH, IMAGE_WIDTH)
        input_shape = (1, IMAGE_WIDTH, IMAGE_WIDTH)
    else:
        x_train = x_train.reshape(x_train.shape[0], IMAGE_WIDTH, IMAGE_WIDTH,
                                  3)
        x_test = x_test.reshape(x_test.shape[0], IMAGE_WIDTH, IMAGE_WIDTH, 3)
        input_shape = (IMAGE_WIDTH, IMAGE_WIDTH, 3)

    # The data, split between train and test sets:
    print('x_train shape:', x_train.shape)
    print(x_train.shape, 'train samples')
    print(x_test.shape, 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('y_test', y_test)

    model = squeezenet.Squeezenet()
    opt = keras.optimizers.SGD()

    # Let's train the model using RMSprop
    model.compile(
        loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=FLAGS.batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_loop_count',
        type=int,
        default=20000,
        help='Number of training iterations')
    parser.add_argument(
        '--batch_size', type=int, default=50, help='Batch Size')
    parser.add_argument(
        '--test_image_count',
        type=int,
        default=None,
        help="Number of test images to evaluate on")
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='''L2 regularization factor for convolution layer weights.
                0.0 indicates no regularization.''')
    parser.add_argument('--batch_norm_decay', type=float, default=0.9)
    FLAGS, unparsed = parser.parse_known_args()
    main()
