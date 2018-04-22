import tensorflow as tf
from ..COMMON.MODEL import Model, BUILD_NET

class DuelingQN(Model):
    def __init__(self,
        action_n,
        shape = [16, 16],
        shape_A = [],
        shape_V = [],
        name="DuelingQ"):
        Model.__init__(self, name=name)
        self.act_n = action_n
        self.hidden_shape = shape
        self.hidden_shape_A = shape_A
        self.hidden_shape_V = shape_V

    
    def __call__(self, obs, train=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            #hidden layer
            net = BUILD_NET(obs, 
                            self.hidden_shape,
                            "HiddenLayer",
                            tf.nn.relu,
                            trainable=train)

            #Advantage layer
            A_net = net
            if len(self.hidden_shape_A):
                A_net = BUILD_NET(A_net, 
                                  self.hidden_shape_A, 
                                  "Hidden_A",
                                  tf.nn.relu,
                                  trainable=train)
            A_net = BUILD_NET(A_net, 
                              [self.act_n],
                              "Advantage",
                              trainable=train)
            A_net = tf.subtract(A_net, tf.reduce_mean(A_net, axis=1, keepdims=True))

            #Value layer
            V_net = net
            if len(self.hidden_shape_V):
                V_net = BUILD_NET(V_net, 
                                  self.hidden_shape_V,
                                  "Hidden_V",
                                  tf.nn.relu,
                                  trainable=train)
            V_net = BUILD_NET(V_net
                              [1],
                              trainable=train)

            Q_Table = tf.add(V_net, A_net, name="Q")

        return Q_Table


