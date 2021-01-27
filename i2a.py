import os
import gym
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected
import tensorflow.contrib.layers as layers
from common.simple import Simple
from common.multiprocessing_env import SubprocVecEnv
from tqdm import tqdm

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

# Hyperparameter of how far ahead in the future the agent "imagines"
# Currently this is specifying one frame in the future.
#NUM_ROLLOUTS = 1

# Hidden size in RNN imagination encoder.
HIDDEN_SIZE = 256

#N_STEPS = 5

# Softmax function for numpy taken from
# https://nolanbconaway.github.io/blog/2017/softmax-numpy
def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

class SimplePolicy(object):
    def __init__(self, ob_space, ac_space, nbatch, nsteps, reuse=False):
        num_rewards = 1
        num_actions = ac_space.n
        num_states = ob_space.n

        with tf.variable_scope('model', reuse=reuse):
            # Model free path.
            self.state = tf.placeholder(tf.float32, [None, num_states])

            state_batch_size = tf.shape(self.state)[0]
            
            h1 = fully_connected(self.state, 256)

            self.logits = linear(h1, num_actions, "action", normalized_columns_initializer(0.01))
            self.vf = tf.reshape(linear(h1, 1, "value", normalized_columns_initializer(1.0)), [-1])

        #A3C style action
        self.sample = categorical_sample(self.logits, num_actions)[0, :]
        
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def act(self, ob):
        sess = tf.get_default_session()
        #imagined_state, imagined_reward, ob = self.transform_input(ob)
        #print("in act : ", imagined_reward.shape)

        a, v = sess.run([
                self.sample,
                self.vf
            ],
            {
                self.state: ob
        })
        return a, v


    def value(self, ob):
        sess = tf.get_default_session()
        #imagined_state, imagined_reward, ob = self.transform_input(ob)
        #print("in value : ", imagined_reward.shape)

        v = sess.run(self.vf, {
            self.state: ob
        })
        return v[0]