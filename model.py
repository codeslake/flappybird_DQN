import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

class Qnetwork():
    def __init__(self, h_size, name, reuse, lr_init=1e-6, beta=0.9):
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None
        with tf.variable_scope(name, reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            self.image = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32)
            self.scalar = tf.placeholder(shape=[None, 7, 4], dtype=tf.float32)

            # image
            n = InputLayer(self.image, name = 'input_image')

            nn = Conv2d(n, 32, (8, 8), (4, 4), act = tf.nn.relu, padding = 'VALID', W_init=w_init, name='n32s4/c')
            nn = Conv2d(nn, 64, (4, 4), (2, 2), act = tf.nn.relu, padding = 'VALID', W_init=w_init, name='n64s2/c')
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act = tf.nn.relu, padding = 'VALID', W_init=w_init, name='n64s1/c')
            nn = Conv2d(nn, h_size, (7, 7), (1, 1), act = tf.nn.relu, padding = 'VALID', W_init=w_init, name='n512s1/c')
            nn_image = FlattenLayer(nn, name = 'flatten_image')

            # scalar
            n = InputLayer(self.scalar, name = 'input_scalar')
            nn = FlattenLayer(n, name = 'flatten_scalar')
            nn = DenseLayer(nn, 32, act = tf.nn.relu, W_init=w_init, b_init=b_init, name = 'n32/fc1/scalar')
            nn = DenseLayer(nn, 64, act = tf.nn.relu, W_init=w_init, b_init=b_init, name = 'n64/fc2/scalar')
            nn_scalar = DenseLayer(nn, 512, act = tf.nn.relu, W_init=w_init, b_init=b_init, name = 'n512/fc3/flatten_scalar')

            # concat
            nn = ConcatLayer([nn_image, nn_scalar], name = 'scalar_input')
            nn = DenseLayer(nn, 1024, act = tf.nn.relu, W_init=w_init, b_init=b_init, name = 'n1024/fc1')
            nn = DenseLayer(nn, 512, act = tf.nn.relu, W_init=w_init, b_init=b_init, name = 'n512/fc2')
            self.Qout = DenseLayer(nn, 2, W_init=w_init, b_init=b_init, name = 'n2/fc3')
            self.Qout = self.Qout.outputs

            self.predict = tf.argmax(self.Qout, 1)

            self.targetQ = tf.placeholder(shape=None, dtype=tf.float32)
            self.actions = tf.placeholder(shape=None, dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, 2, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
            self.learning_rate = tf.Variable(lr_init, trainable=False)
            self.trainer = tf.train.AdamOptimizer(self.learning_rate, beta1=beta)
            self.updateModel = self.trainer.minimize(self.loss)
