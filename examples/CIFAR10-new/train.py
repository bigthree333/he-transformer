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
from keras.models import model_from_json
from keras import losses

import numpy as np
import squeezenet
import tensorflow as tf
import time
from datetime import datetime
import os
import argparse
import sys

FLAGS = None


def main():
    # Import data
    num_classes = 10
    NUM_TRAIN_SAMPLES = 50000
    IMAGE_HEIGHT = IMAGE_WIDTH = 32
    CHANNELS = 3

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

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
    def loss(y_true, y_pred):
        return tf.keras.backend.categorical_crossentropy(
            y_true, y_pred, from_logits=True)

    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        verbose=1,
        validation_data=(x_test, y_test))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    convert_model_to_tf()


def convert_model_to_tf():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    K.set_learning_phase(0)

    model = loaded_model

    print('inputs', model.inputs)
    print('outputs', model.outputs)

    def freeze_session(session,
                       keep_var_names=None,
                       output_names=None,
                       clear_devices=True):
        """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                            or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(
                set(v.op.name for v in tf.global_variables()).difference(
                    keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            # Graph -> GraphDef ProtoBuf
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph

    frozen_graph = freeze_session(
        K.get_session(),
        output_names=[out.op.name for out in loaded_model.outputs])

    tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs', type=int, default=1, help='Number of training iterations')
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
