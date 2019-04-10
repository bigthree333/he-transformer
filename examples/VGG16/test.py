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

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

FLAGS = None

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model


def optimize_for_inference():
    K.set_learning_phase(0)
    model = VGG16()
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
            graphdef_inf = tf.compat.v1.graph_util.remove_training_nodes(
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

    os.system('''python -m tensorflow.python.tools.optimize_for_inference \
        --input model/frozen_model.pb \
        --input_names input_1 \
        --output_names predictions/Softmax \
        --output model/optimized_model.pb''')
    print('optimized for inference')


def main():
    if True:
        optimize_for_inference()
        tf.keras.backend.clear_session()

        f = gfile.FastGFile("./model/frozen_model.pb", 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        f.close()
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        sess.graph.as_default()
        tf.import_graph_def(graph_def)

        nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        print('nodes', nodes)

        x_test = np.random.random([FLAGS.batch_size, 224, 224, 3])
        x_test = preprocess_input(x_test)

        output_tensor = sess.graph.get_tensor_by_name(
            'import/predictions/Softmax:0')
        input_tensor = sess.graph.get_tensor_by_name('import/input_1:0')
        print('input_tensor', input_tensor)
        print('output tensor', output_tensor.shape)

        preds = sess.run(output_tensor,
                         {input_tensor: x_test[:FLAGS.batch_size]})
        preds = np.argmax(preds, axis=1)

        print('preds', preds)

        exit(1)

    # Original VGG16 Keras model
    x = np.random.random([FLAGS.batch_size, 224, 224, 3])
    x = preprocess_input(x)
    K.set_learning_phase(0)
    model = VGG16()
    model.summary()
    test = [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

    session = keras.backend.get_session()
    min_graph = tf.graph_util.convert_variables_to_constants(
        session, session.graph_def, [node.op.name for node in model.outputs])

    tf.train.write_graph(min_graph, "./model/", "model.pbtxt", as_text=True)

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
