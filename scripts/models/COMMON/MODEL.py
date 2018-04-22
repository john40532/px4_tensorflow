import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


def BUILD_NET(input_var, 
              hidden_wide,
              name,
              activation = None,
              weight_initializer = tf.random_normal_initializer(0, 0.1),
              bias_initializer = tf.constant_initializer(0.),
              trainable = True):
  assert type(hidden_wide) == type([])

  layer = tf.layers.dense(input_var, 
                          hidden_wide[0],
                          activation,
                          kernel_initializer = weight_initializer,
                          bias_initializer = bias_initializer,
                          trainable = trainable,
                          name = name+"0")
  for i in range(len(hidden_wide)-1):
    layer = tf.layers.dense(layer,
                            hidden_wide[i+1],
                            activation,
                            kernel_initializer = weight_initializer,
                            bias_initializer = bias_initializer,
                            trainable = trainable,
                            name = name+"{}".format(i+1))
  return layer

