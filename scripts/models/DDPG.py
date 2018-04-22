from copy import copy
import tensorflow as tf
from .COMMON import MEMORY as MEM

class DDPG:
    def __init__(self, 
                 actor, 
                 critic, 
                 observation_shape, 
                 action_shape,
                 gamma = 0.99,
                 tau = 0.001, 
                 memory_size = 2**14,
                 batch_size = 128,
                 actor_lr = 1e-4,
                 critic_lr = 1e-3):
        self.obs0 = tf.placeholder(tf.float32, (None,)+observation_shape, name = "obs0")
        self.obs1 = tf.placeholder(tf.float32, (None,)+observation_shape, name = "obs1")
        self.term = tf.placeholder(tf.float32, (None, 1)                , name = "term")
        self.rewards = tf.placeholder(tf.float32, (None, 1)             , name = "rewards")

        self.gamma = tf.constant(gamma, name = "gamma")
        self.tau   = tf.constant(tau,   name = "tau")
        self.memory = MEM.Memory(observation_shape[0], action_shape[0], memory_size, batch_size)
        self.actor = actor
        self.critic = critic
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        target_actor = copy(actor)
        target_actor.name = "Target_actor"
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = "Target_critic"
        self.target_critic = target_critic

        self.actor_tf = actor(self.obs0)
        self.critic_with_actor_tf = critic(self.obs0, self.actor_tf)
    
        Q_obs1 = target_critic(self.obs1, target_actor(self.obs1, train=False), train=False)

        with tf.name_scope("Update"):
            act_vars = self.actor.vars
            cri_vars = self.critic.vars
            tar_act_vars = self.target_actor.vars
            tar_cri_vars = self.target_critic.vars
            assert len(act_vars) == len(tar_act_vars)
            assert len(self.critic.vars) == len(self.target_critic.vars)
            actor_soft = [tf.assign(t, (1-tau)*t + tau*e) for t,e in zip(tar_act_vars, act_vars)]
            actor_init = [tf.assign(t, e) for t,e in zip(tar_act_vars, act_vars)]
            critic_soft = [tf.assign(t ,(1-tau)*t + tau*e) for t,e, in zip(tar_cri_vars, cri_vars)]
            critic_init = [tf.assign(t ,e) for t,e, in zip(tar_cri_vars, cri_vars)]

            #TODO: manage log
            print("Setting update...")
            for t,e in zip(tar_act_vars, act_vars):
                print("{}<----{}".format(t.name, e.name))
            for t,e in zip(tar_cri_vars, cri_vars):
                print("{}<----{}".format(t.name, e.name))

            soft_update = actor_soft + critic_soft
            self.soft_update = tf.group(*soft_update)
            self.init_update = critic_init + actor_init
   
        
        with tf.name_scope("Loss"):
            with tf.name_scope("TargetQ"):
                self.targetQ = tf.add(self.rewards, (1-self.term)*self.gamma*Q_obs1)

            with tf.name_scope("Actor"):
                self.actor_loss = tf.reduce_mean(-self.critic_with_actor_tf)

            with tf.name_scope("Critic"):
                #targetQ = tf.stop_gradient(self.targetQ)
                self.critic_loss = tf.losses.mean_squared_error(labels=self.targetQ, predictions=self.critic_with_actor_tf)
       
        with tf.variable_scope("Learning"):
            self.criticTraining = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss, var_list=self.critic.trainable_vars)
            self.actorTraining = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss, var_list=self.actor.trainable_vars)

    def training_tf(self, sess):
        idxs = self.memory.random_idxs()
        states, actions, rewards, state_s, terms = self.memory.batch(idxs)
        sess.run(self.actorTraining, {self.obs0: states})
        sess.run(self.criticTraining, {self.obs0: states,
                                       self.obs1: state_s,
                                       self.actor_tf: actions,
                                       self.term: terms.reshape(-1,1),
                                       self.rewards: rewards.reshape(-1,1)})
        sess.run(self.soft_update)

    def store(self, s, a, r, s_, done):
        self.memory.store(s,a,r,s_,done)
