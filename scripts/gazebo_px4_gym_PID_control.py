#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import rospy
import roslaunch
import time, sys, argparse, math
import random
import numpy as np
from pymavlink import mavutil
from rosgraph_msgs.msg import Clock
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from Quadcopter_simulator.controller import Controller_PID_Point2Point

# Set goals to go to
GOALS = [(1,1,2),(1,-1,4),(-1,-1,2),(-1,1,4)]
YAWS = [0,3.14,-1.54,1.54]
# Define the quadcopters
QUADCOPTER={'q1':{'position':[1,0,4],'orientation':[0,0,0],'L':0.3,'r':0.1,'prop_size':[10,4.5],'weight':1.2}}
# Controller parameters
CONTROLLER_PARAMETERS = {'Motor_limits':[10,1100],
                    'Tilt_limits':[-30,30],
                    'Yaw_Control_Limits':[-900,900],
                    'Z_XY_offset':5000,
                    'Linear_PID':{'P':[1,1,1500],'I':[0,0,100],'D':[0,0,500]},
                    'Linear_To_Angular_Scaler':[10,10,0],
                    'Yaw_Rate_Scaler':0.18,
                    'Angular_PID':{'P':[6,6,70],'I':[10,10,1.2],'D':[100,100,0]},
                    }

# Make objects for quadcopter, gui and controller

ctrl = Controller_PID_Point2Point(params=CONTROLLER_PARAMETERS)



class World:
    def __init__(self):
        self.iris_index = 2
        self.position_x = 0
        self.position_y = 0
        self.position_z = 0
        self.orientation_x = 0
        self.orientation_y = 0
        self.orientation_z = 0
        self.orientation_w = 1
        self.linear_x = 0
        self.linear_y = 0
        self.linear_z = 0
        self.angular_x = 0
        self.angular_y = 0
        self.angular_z = 0
        self.random_seed = None

    def call_next_step_srv(self):
        rospy.wait_for_service('/gazebo/next_step')
        try:
            pause = rospy.ServiceProxy('/gazebo/next_step', Empty)
            return pause()
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def seed(self, seed_num):
        self.random_seed = seed_num


    def randQ(self):
        sigma1 = math.sqrt(1.0 - random.random())
        sigma2 = math.sqrt(random.random())
        theta1 = 2*math.pi*random.random()
        theta2 = 2*math.pi*random.random()
        w = math.cos(theta2)*sigma2
        x = math.sin(theta1)*sigma1
        y = math.cos(theta1)*sigma1
        z = math.sin(theta2)*sigma2
        return w, x, y, z

    def reset(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            reset_world_handle = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            reset_world_handle()
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

        init_model_state = ModelState()
        if self.random_seed != None:
            if self.random_seed != 0:
                random.seed(self.random_seed)
            init_model_state.model_name = 'iris'
            init_model_state.pose.position.x = random.uniform(-2,2)
            init_model_state.pose.position.y = random.uniform(-2,2)
            init_model_state.pose.position.z = random.uniform(4,6)
            w, x, y, z = self.randQ()
            init_model_state.pose.orientation.x = x
            init_model_state.pose.orientation.y = y
            init_model_state.pose.orientation.z = z
            init_model_state.pose.orientation.w = w
            init_model_state.twist.linear.x = random.uniform(-1,1)
            init_model_state.twist.linear.y = random.uniform(-1,1)
            init_model_state.twist.linear.z = random.uniform(-1,1)
            init_model_state.twist.angular.x = random.uniform(-1,1)
            init_model_state.twist.angular.y = random.uniform(-1,1)
            init_model_state.twist.angular.z = random.uniform(-1,1)
            init_model_state.reference_frame = 'world'

        else:
            init_model_state.model_name = 'iris'
            init_model_state.pose.position.x = 0
            init_model_state.pose.position.y = 0
            init_model_state.pose.position.z = 4
            init_model_state.pose.orientation.x = 0
            init_model_state.pose.orientation.y = 0
            init_model_state.pose.orientation.z = 0
            init_model_state.pose.orientation.w = 1
            init_model_state.twist.linear.x = 0
            init_model_state.twist.linear.y = 0
            init_model_state.twist.linear.z = 0
            init_model_state.twist.angular.x = 0
            init_model_state.twist.angular.y = 0
            init_model_state.twist.angular.z = 0
            init_model_state.reference_frame = 'world'


        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_model_state_handle = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_model_state_handle(init_model_state)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

        return [init_model_state.pose.position.x, init_model_state.pose.position.y, init_model_state.pose.position.z,
                 init_model_state.pose.orientation.x, init_model_state.pose.orientation.y, init_model_state.pose.orientation.z, init_model_state.pose.orientation.w,
                 init_model_state.twist.linear.x, init_model_state.twist.linear.y, init_model_state.twist.linear.z, 
                 init_model_state.twist.angular.x, init_model_state.twist.angular.y, init_model_state.twist.angular.z]

    def delay(self):
        i = 100000000
        while i > 0:
            i -= 1

    def model_states_callback(self, data):
        self.position_x = data.pose[self.iris_index].position.x
        self.position_y = data.pose[self.iris_index].position.y
        self.position_z = data.pose[self.iris_index].position.z
        self.orientation_x = data.pose[self.iris_index].orientation.x
        self.orientation_y = data.pose[self.iris_index].orientation.y
        self.orientation_z = data.pose[self.iris_index].orientation.z
        self.orientation_w = data.pose[self.iris_index].orientation.w
        self.linear_x = data.twist[self.iris_index].linear.x
        self.linear_y = data.twist[self.iris_index].linear.y
        self.linear_z = data.twist[self.iris_index].linear.z
        self.angular_x = data.twist[self.iris_index].angular.x
        self.angular_y = data.twist[self.iris_index].angular.y
        self.angular_z = data.twist[self.iris_index].angular.z


    def step(self, actuator_cmd):

        assert len(actuator_cmd) is 8

        udp.mav.rc_channels_override_send(udp.target_system, udp.target_component, 
                                          actuator_cmd[0],
                                          actuator_cmd[1],
                                          actuator_cmd[2],
                                          actuator_cmd[3],
                                          actuator_cmd[4],
                                          actuator_cmd[5],
                                          actuator_cmd[6],
                                          actuator_cmd[7],
                                          )
        udp.recv_msg()
        self.call_next_step_srv()

        state = [self.position_x, self.position_y,self.position_z,
                 self.orientation_x,self.orientation_y,self.orientation_z,self.orientation_w,
                 self.linear_x, self.linear_y, self.linear_z, 
                 self.angular_x, self.angular_y, self.angular_z]

        distance = math.sqrt((self.position_x)**2 + (self.position_y)**2 + (self.position_z-5)**2)
        reward = math.exp(distance);

        done = False
        if distance > 3:
            done = True

        return state, reward, done, Empty

if __name__ == '__main__':
    print "Connecting"
    udp = mavutil.mavudp('127.0.0.1:14560', source_system=0)  #gazebo
    rospy.init_node('listener', anonymous=True)
    rate = rospy.Rate(500) # 10hz
    env = World()
    # env.seed(0)
    rospy.Subscriber("/gazebo/model_states", ModelStates, env.model_states_callback)
    while not rospy.is_shutdown():
        prev_states = env.reset()
        target = [1, 1, 4]
        for i in range(5000):
            actions = ctrl.update(target, prev_states)
            a_extend = np.concatenate((actions, np.zeros(4, dtype=int)))
            states, reward, done, info = env.step(a_extend)
            # states, reward, done, info = env.step([0, 0, 0, 0 ,0,0,0,0])
            prev_states = states
            # if i % 10 == 0:
            #     print a_extend
