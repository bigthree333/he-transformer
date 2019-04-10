import keras
from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import argparse
import os
import ngraph_bridge
# CPU needs NHWC format for MaxPool / FusedBatchNorm
keras.backend.set_image_data_format('channels_last')

from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

FLAGS = None


def optimize_for_inference():
    os.system('''python -m tensorflow.python.tools.optimize_for_inference \
        --input model/tf_model.pb \
        --output model/tf_optimized_model.pb''')
    print('optimized for inference')


def main():
    optimize_for_inference()

    NUM_TRAIN_SAMPLES = 50000
    IMAGE_HEIGHT = IMAGE_WIDTH = 32
    CHANNELS = 3
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    #if K.image_data_format() == 'channels_first':
    #    x_train = x_train.reshape(x_train.shape[0], 3, IMAGE_HEIGHT,
    #                              IMAGE_WIDTH)
    #    x_test = x_test.reshape(x_test.shape[0], 3, IMAGE_WIDTH, IMAGE_WIDTH)
    #    input_shape = (1, IMAGE_WIDTH, IMAGE_WIDTH)
    #else:
    #    x_train = x_train.reshape(x_train.shape[0], IMAGE_WIDTH, IMAGE_WIDTH,
    #                              3)
    #    x_test = x_test.reshape(x_test.shape[0], IMAGE_WIDTH, IMAGE_WIDTH, 3)
    #    input_shape = (IMAGE_WIDTH, IMAGE_WIDTH, 3)

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

    f = gfile.FastGFile("./model/tf_optimized_model.pb", 'rb')
    graph_def = tf.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    sess.graph.as_default()
    # Import a serialized TensorFlow `GraphDef` protocol buffer
    # and place into the current default `Graph`.
    tf.import_graph_def(graph_def)

    nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    print('nodes', nodes)

    output_tensor = sess.graph.get_tensor_by_name('import/output_1/BiasAdd:0')
    input_tensor = sess.graph.get_tensor_by_name('import/input_1:0')
    print('input_tensor', input_tensor)
    print('x_test shape', x_test[:FLAGS.batch_size].shape)
    print('output tensor', output_tensor.shape)

    preds = sess.run(output_tensor, {input_tensor: x_test[:FLAGS.batch_size]})
    preds = np.argmax(preds, axis=1)

    y_test_batch = y_test[:FLAGS.batch_size]

    y_test_batch = np.argmax(y_test_batch, axis=1)
    print(y_test_batch)

    print('preds', preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=50, help='Batch Size')
    FLAGS, unparsed = parser.parse_known_args()
    main()
