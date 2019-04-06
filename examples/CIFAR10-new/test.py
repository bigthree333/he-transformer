import keras
from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import ngraph_bridge
# CPU needs NHWC format for MaxPool / FusedBatchNorm
keras.backend.set_image_data_format('channels_last')

from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

NUM_TRAIN_SAMPLES = 50000
IMAGE_HEIGHT = IMAGE_WIDTH = 32
CHANNELS = 3
num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    x_test = x_test.reshape(x_test.shape[0], 3, IMAGE_WIDTH, IMAGE_WIDTH)
    input_shape = (1, IMAGE_WIDTH, IMAGE_WIDTH)
else:
    x_train = x_train.reshape(x_train.shape[0], IMAGE_WIDTH, IMAGE_WIDTH, 3)
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
if False:
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    model = loaded_model

    print(model.outputs)
    # [<tf.Tensor 'dense_2/Softmax:0' shape=(?, 10) dtype=float32>]
    print(model.inputs)

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

f = gfile.FastGFile("./model/tf_model.pb", 'rb')
graph_def = tf.GraphDef()
# Parses a serialized binary message into the current message.
graph_def.ParseFromString(f.read())
f.close()

TEST_SIZE = 1

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

sess.graph.as_default()
# Import a serialized TensorFlow `GraphDef` protocol buffer
# and place into the current default `Graph`.
tf.import_graph_def(graph_def)

nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
print(nodes)

softmax_tensor = sess.graph.get_tensor_by_name('import/dense_2/Softmax:0')

preds = sess.run(softmax_tensor, {'import/input_1:0': x_test[:TEST_SIZE]})
preds = np.argmax(preds, axis=1)

y_test_batch = y_test[:TEST_SIZE]

y_test_batch = np.argmax(y_test_batch, axis=1)
print(y_test_batch)

print('preds', preds)