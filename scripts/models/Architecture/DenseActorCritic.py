import tensorflow as tf
from ..COMMON.MODEL import Model, BUILD_NET

class Actor(Model):
    def __init__(self, 
                 act_dim,
                 act_bound,
                 hidden_shape=[64,64], 
                 name="Actor"):
        Model.__init__(self, name=name)
        self.act_dim = act_dim
        self.hidden_shape = hidden_shape
        self.act_bound = act_bound

    def __call__(self, obs, train=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            actor = BUILD_NET(obs, 
                              self.hidden_shape, 
                               "HiddenLayer", 
                               tf.nn.relu, 
                               trainable=train)
            actor = BUILD_NET(actor,
                              [self.act_dim],
                              "Action_Output",
                              tf.nn.tanh,
                              trainable=train)
            actor = tf.multiply(actor, self.act_bound, name="Scaled_Action")
        return actor

class Critic(Model):
    def __init__(self,
                 hidden_shape=[64,64],
                 name="Critic"):
        Model.__init__(self, name=name)
        self.hidden_shape = hidden_shape

    def __call__(self, obs, act, train=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            featured = tf.concat([obs, act], axis=-1)
            critic = BUILD_NET(featured, 
                               self.hidden_shape, 
                               "HiddenLayer", 
                               tf.nn.relu, 
                               trainable=train)
 
            critic = BUILD_NET(critic, 
                               [1], 
                               "value", 
                               trainable=train)
        return critic

