import tensorflow as tf
import numpy as np
import gym
import os
import json
from datetime import datetime

import models.Architecture.DenseActorCritic as AC
from models.DDPG import DDPG as ddpg

TASK = {
    "model"               : "DDPG",
    "ENV_NAME"            : "Pendulum-v0",
    "actor"               : [32,32],
    "critic"              : [32,32],
    "actor learning rate" : 0.01,
    "critic learning rate": 0.02,
    "gamma"               : .9,
    "tau"                 : .01,
    "memory capacity"     : 10000,
    "batch size"          : 32,
    "MAX_EPI"             : 500
        }

def ACTION_NOISE(MAX_EPI):
    alpha = -10*np.log(2)/MAX_EPI
    c = 0.0001
    for epi in range(MAX_EPI):
        yield epi, (1-c)*np.exp(alpha*epi)+c


def TRAIN(TASK_DICT):
    tf.reset_default_graph()
    now = datetime.now()
    date = now.strftime('%Y%m%d_%H%M')
    LOGDIR = "./_logs/{}/{}_{}".format(TASK_DICT["ENV_NAME"], 
                                       TASK_DICT["model"],
                                       date)
    TMPDIR = "./_tmp/model/{}_{}/".format(TASK_DICT["ENV_NAME"],
                                          date)
    if not os.path.isdir(TMPDIR):
        os.makedirs(TMPDIR)

    with open(TMPDIR+"log", "w") as f:
        f.write(json.dumps(TASK_DICT, sort_keys=True, indent=4))

    env = gym.make(TASK_DICT["ENV_NAME"])

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    actor  = AC.Actor(act_dim, a_bound, TASK_DICT["actor"])
    critic = AC.Critic(TASK_DICT["critic"])
    agent  = ddpg(actor, critic, (obs_dim,), (act_dim,),
                  gamma      =TASK_DICT["gamma"],
                  tau        =TASK_DICT["tau"],
                  memory_size=TASK_DICT["memory capacity"],
                  batch_size =TASK_DICT["batch size"],
                  actor_lr   =TASK_DICT["actor learning rate"],
                  critic_lr  =TASK_DICT["critic learning rate"])
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(v.name, v)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        Writer = tf.summary.FileWriter(LOGDIR, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(agent.init_update)

        print("Start Training")
        for epi, var in ACTION_NOISE(TASK_DICT["MAX_EPI"]):
            s = env.reset()
            done = False
            R = 0
            while not done:
                env.render()
                if agent.memory.Full:
                    agent.training_tf(sess)

                if np.random.random()>var:
                    a = sess.run(agent.actor_tf, {agent.obs0:[s]})[0]
                    a = np.clip(np.random.normal(a, var/10), -a_bound, a_bound)
                else:
                    a = env.action_space.sample()

                s_, r, done, info = env.step(a)
                R+=r
                agent.memory.store(s,a,r,s_,done)
                s=s_
            summary = sess.run(merged)
            Writer.add_summary(summary, epi)
            summ = tf.Summary(value=[tf.Summary.Value(tag="Score", simple_value=R)])
            Writer.add_summary(summ, epi)
            summ = tf.Summary(value=[tf.Summary.Value(tag="var", simple_value=var)])
            Writer.add_summary(summ, epi)
            print("EPI:{},\tScore:{},\tVAR:{}".format(epi, R, var))
    save = tf.train.Saver()
    save.save(sess, TMPDIR+"DDPG")

TRAIN(TASK)
