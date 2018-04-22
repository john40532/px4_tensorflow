import tensorflow as tf
from ..COMMON.MODEL import Model, BUILD_NET

class QN(Model):
    def __init__(self,
        action_n,
        shape = [16, 16],
        name="Q_Net"
    ):
        Model.__init__(self, name=name)
        self.act_n = action_n
        self.hidden_shape = shape

    
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

            Q_Table = BUILD_NET(net,
                                [self.act_n],
                                "Q",
                                trainable=train)

        return Q_Table


