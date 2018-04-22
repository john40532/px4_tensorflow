from copy import copy
import tensorflow as tf
from .COMMON import MEMORY as MEM

class QLearning:
    def __init__(self,
                 QNet,
                 observation_shape,
                 action_n,
                 gamma = 0.9,
                 memory_size = 2**14,
                 batch_size = 128,
                 learning_rate = 1e-3
    ):
        self.obs0 = tf.placeholder(tf.float32, (None,)+observation_shape, name = "obs0")
        self.obs1 = tf.placeholder(tf.float32, (None,)+observation_shape, name = "obs1")

        self.gamma = tf.constant(gamma, name = "gamma")
        self.memory = MEM.Memory(observation_shape[0], 1, memory_size, batch_size)
        self.QNet = QNet
        self.lr = learning_rate

        target_QNet = copy(QNet)
        target_QNet.name = "Target_qNet"
        self.target_QNet = target_QNet

        #build Q-Network
        self.QNet_tf = QNet(self.obs0)
        self.qOptimal = tf.argmax(self.QNet_tf, 1, "Action")

        #target Q-Network
        Q_obs1 = target_QNet(self.obs1, train=False)

        with tf.name_scope("Update"):
            Q_vars     = self.QNet.vars
            tar_Q_vars = self.target_QNet.vars
            assert len(Q_vars) == len(tar_Q_vars)
            Q_update = [tf.assign(t,e) for t,e in zip(tar_Q_vars, Q_vars)]
            self.update = tf.group(*Q_update)

        with tf.name_scope("Experience"):
            self.term = tf.placeholder(tf.float32, (None,1)        , name = "term")
            self.action = tf.placeholder(tf.int32, (None)        , name = "action")
            self.opt_action = tf.placeholder(tf.int32, (None)    , name = "action_opt")
            self.rewards = tf.placeholder(tf.float32, (None,1)     , name = "rewards")
            with tf.name_scope("Action_Mask"):
                optimal_action_mask = tf.one_hot(self.opt_action, action_n, 1., 0., name = "optAct_mask")
                action_mask = tf.one_hot(self.action, action_n, 1., 0., name = "Act_mask")
                inv_action_mask = tf.one_hot(self.action, action_n, 0., 1., name = "invAct_mask")

        with tf.name_scope("TD_ERROR"):
            tMax = tf.reduce_sum(Q_obs1*optimal_action_mask,1,keepdims=True)
            targetQ = self.rewards+self.gamma*tMax*(tf.constant(1.)-self.term)
            targetQ = action_mask*targetQ + inv_action_mask*self.QNet_tf
            
            self.TD_err = tf.reduce_mean(tf.squared_difference(targetQ, self.QNet_tf))

        with tf.variable_scope("Learning"):
            self.train = tf.train.RMSPropOptimizer(self.lr).minimize(self.TD_err)

    def store(self, s, a, r, s_, done):
        self.memory.store(s,a,r,s_,done)

    def training_tf(self, sess):
        idxs = self.memory.random_idxs()
        states, actions, rewards, state_s, terms = self.memory.batch(idxs)
        opt_action = sess.run(self.qOptimal, {self.obs0: state_s}).reshape([-1])
        sess.run(self.train, {self.obs0: states,
                              self.obs1: state_s,
                              self.action: actions.reshape([-1]),
                              self.opt_action: opt_action.reshape([-1]),
                              self.rewards: rewards.reshape((-1,1)),
                              self.term: terms.reshape((-1,1))})

