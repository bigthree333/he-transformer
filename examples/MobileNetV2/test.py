import keras
from keras.models import model_from_json
from keras import backend as K
import numpy as np
import argparse
import os
import ngraph_bridge
# CPU needs NHWC format for MaxPool / FusedBatchNorm
keras.backend.set_image_data_format('channels_last')

from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input, decode_predictions

FLAGS = None

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model

#def optimize_for_inference():
#    os.system('''python -m tensorflow.python.tools.optimize_for_inference \
#        --input model/tf_model.pb \
#        --output model/tf_optimized_model.pb''')
#    print('optimized for inference')


def optimize_for_inference():
    K.set_learning_phase(0)
    model = MobileNetV2()
    K.set_learning_phase(0)
    model_file = './model/model.h5'
    model.save(model_file)

    # Clear any previous session.
    tf.keras.backend.clear_session()

    save_pb_dir = './model'
    model_fname = './model/model.h5'

    def freeze_graph(graph,
                     session,
                     output,
                     save_pb_dir='.',
                     save_pb_name='frozen_model.pb',
                     save_pb_as_text=False):
        with graph.as_default():
            graphdef_inf = tf.graph_util.remove_training_nodes(
                graph.as_graph_def())
            graphdef_frozen = tf.graph_util.convert_variables_to_constants(
                session, graphdef_inf, output)
            graph_io.write_graph(
                graphdef_frozen,
                save_pb_dir,
                save_pb_name,
                as_text=save_pb_as_text)
            return graphdef_frozen

    # This line must be executed before loading Keras model.
    tf.keras.backend.set_learning_phase(0)

    model = load_model(model_fname)

    session = tf.keras.backend.get_session()

    INPUT_NODE = [t.op.name for t in model.inputs]
    OUTPUT_NODE = [t.op.name for t in model.outputs]
    print(INPUT_NODE, OUTPUT_NODE)
    frozen_graph = freeze_graph(
        session.graph,
        session, [out.op.name for out in model.outputs],
        save_pb_dir=save_pb_dir)


def main():
    #optimize_for_inference()

    img = np.random.random([FLAGS.batch_size, 224, 224, 3])

    # model = ResNet50(weights='imagenet')
    #img_path = 'elephant.jpg'
    #img = image.load_img(img_path, target_size=(224, 224))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    x = img
    x = preprocess_input(x)

    K.set_learning_phase(0)

    model = MobileNetV2()
    model.summary()

    preds = model.predict(x)

    print('pred', decode_predictions(preds, top=3)[0])

    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=50, help='Batch Size')
    FLAGS, unparsed = parser.parse_known_args()
    main()
