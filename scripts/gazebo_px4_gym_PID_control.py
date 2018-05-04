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
import csv
import os
import sys
from pymavlink import mavutil
from Quadcopter_simulator.controller import Controller_PID_Point2Point
from gazebo_px4_gym_ros import World as world

# Controller parameters
CONTROLLER_PARAMETERS = {'Motor_limits':[10,1100],
                    'Tilt_limits':[-10,10],
                    'Yaw_Control_Limits':[-500,500],
                    'Z_XY_offset':5000,
                    'Linear_PID':{'P':[0.2,0.2,1200],'I':[0.01,0.01,100],'D':[0.02,0.02,500]},
                    'Linear_To_Angular_Scaler':[1,1,0],
                    'Yaw_Rate_Scaler':0.5,
                    'Angular_PID':{'P':[500,500,10],'I':[20,20,10],'D':[100,100,0]},
                    }


# Make objects for quadcopter, gui and controller

ctrl = Controller_PID_Point2Point(params=CONTROLLER_PARAMETERS)

env = world()
env.seed(0)
if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    TMPDIR = dir_path + "/_tmp/training_sample/"
    if not os.path.isdir(TMPDIR):
        os.makedirs(TMPDIR)
    file = open(TMPDIR+"sample.csv", "w")
    csvCursor = csv.writer(file)

    # write header to csv file
    csvHeader = ['s', 'a', 'r', 's_next', 'done']
    csvCursor.writerow(csvHeader)

    while True:
        s = env.reset()
        done = False
        R = 0
        while not done:
            a = ctrl.updatePD(env.target, s)
            a_extend = np.concatenate((a, np.zeros(4, dtype=int)))
            s_next, r, done, info = env.step(a_extend)
            R+=r
            s=s_next
            log_data = [s, a, r, s_next, done]
            csvCursor.writerow(log_data)
