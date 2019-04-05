import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d
from tensorflow.contrib.layers import batch_norm, l2_regularizer
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope


def get_variable(name, shape, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()

    if mode == 'train':
        return tf.get_variable(name, shape)
    else:
        return tf.constant(
            np.loadtxt(name + '.txt', dtype=np.float32).reshape(shape))


def fire_module(inputs, squeeze_depth, expand_depth, reuse=None, scope=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with arg_scope([conv2d, max_pool2d]):
            net = _squeeze(inputs, squeeze_depth)
            net = _expand(net, expand_depth)
        return net


def _squeeze(inputs, num_outputs):
    # v1.0 & v1.1
    return conv2d(
        inputs,
        num_outputs, [1, 1],
        stride=1,
        data_format='NHWC',
        scope='squeeze')


def _expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = conv2d(
            inputs,
            num_outputs, [1, 1],
            stride=1,
            data_format='NHWC',
            scope='1x1')
        e3x3 = conv2d(
            inputs, num_outputs, [3, 3], data_format='NHWC', scope='3x3')
    ret = tf.concat([e1x1, e3x3], 3)
    return ret


class Squeezenet_CIFAR(object):
    """Modified version of squeezenet for CIFAR images"""
    name = 'squeezenet_cifar'

    def __init__(self, args):
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False

    def build(self, x, is_training):
        self._is_built = True
        with tf.variable_scope(self.name, values=[x]):
            with arg_scope(
                    _arg_scope(is_training, self._weight_decay,
                               self._batch_norm_decay)):
                return self._squeezenet(x)

    @staticmethod
    def _squeezenet(images, num_classes=10):
        # Input shape N x 32 x 32 x 3
        print('images', images)

        # N x 32 x 32 x 64
        net = conv2d(
            images,
            num_outputs=64,
            kernel_size=[2, 2],
            stride=1,
            data_format='NHWC',
            scope='conv1')
        print('conv1', net)

        # N x 16 x 16 x 64
        net = max_pool2d(net, [2, 2], data_format='NHWC', scope='maxpool1')
        print('maxpool1', net)

        # N x 16 x 16 x 128
        net = fire_module(net, 16, 64, scope='fire2')
        print('fire2', net)

        # N x 16 x 16 x 128
        net = fire_module(net, 16, 64, scope='fire3')
        print('fire3', net)

        #net = fire_module(net, 32, 128, scope='fire4')
        #print('fire4', net)

        # N x 8 x 8 x 128
        net = max_pool2d(net, [2, 2], data_format='NHWC', scope='maxpool4')
        print('maxpool4', net)

        # N x 8 x 8 x 256
        net = fire_module(net, 32, 128, scope='fire5')
        print('fire5', net)

        # N x 8 x 8 x 256
        net = fire_module(net, 48, 192, scope='fire6')
        print('fire6', net)

        #net = fire_module(net, 48, 192, scope='fire7')
        #print('fire7', net)
        #net = fire_module(net, 64, 256, scope='fire8')
        #net = max_pool2d(net, [2, 2], data_format='NHWC', scope='maxpool8')
        #net = fire_module(net, 64, 256, scope='fire9')
        #net = avg_pool2d(net, [4, 4], data_format='NHWC', scope='avgpool10')

        # N x 8 x 8 x 10
        net = conv2d(
            net,
            num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            data_format='NHWC',
            scope='conv10')
        print('conv10', net)

        # N x 1 x 1 x 10
        net = max_pool2d(net, [8, 8], data_format='NHWC', scope='maxpool7')
        print('maxpool7', net)
        # N x 10
        logits = tf.squeeze(net, [1, 2], name='logits')
        print('logits', logits)
        return logits


def _arg_scope(is_training, weight_decay, bn_decay):
    with arg_scope([conv2d],
                   weights_regularizer=l2_regularizer(weight_decay),
                   normalizer_fn=batch_norm,
                   normalizer_params={
                       'is_training': is_training,
                       'fused': True,
                       'decay': bn_decay
                   }):
        with arg_scope([conv2d, avg_pool2d, max_pool2d, batch_norm],
                       data_format='NHWC') as sc:
            return sc
