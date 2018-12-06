# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import math
import sys
import time

import numpy as np
import tensorflow as tf

import cifar10

FLAGS = None

import time

def save_weights():
  """Saves CIFAR10 weights"""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    images = images[0:10]
    labels = labels[0:10]

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.he_inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
      print('Creating session')


      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return

      # Save variables
      for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        weight = (sess.run([var]))[0].flatten().tolist()
        filename = (str(var).split())[1].replace('/', '_')
        filename = 'weights/' + filename.replace("'", "").replace(':0', '') + '.txt'

        print("saving", filename)
        np.savetxt(str(filename), weight)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  save_weights()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--eval_dir',
      type=str,
      default='/tmp/cifar10_eval',
      help='Directory where to write event logs.')
  parser.add_argument(
      '--eval_data',
      type=str,
      default='test',
      help="""Either 'test' or 'train_eval'.""")
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='/tmp/cifar10_train',
      help="""Directory where to read model checkpoints.""")
  parser.add_argument(
      '--eval_interval_secs',
      type=int,
      default=60 * 5,
      help='How often to run the eval.')
  parser.add_argument(
      '--num_examples',
      type=int,
      default=10000,
      help='Number of examples to run.')
  parser.add_argument(
      '--run_once',
      type=bool,
      default=True,
      help='Whether to run eval only once.')


  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)