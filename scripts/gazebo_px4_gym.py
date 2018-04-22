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
from pymavlink import mavutil
import time, sys, argparse, math
from rosgraph_msgs.msg import Clock
from mavros_msgs.msg import ActuatorControl
from mavros_msgs.srv import CommandBool, SetMode
from std_srvs.srv import Empty


def next_step():
    rospy.wait_for_service('/gazebo/next_step')
    try:
        pause = rospy.ServiceProxy('/gazebo/next_step', Empty)
        return pause()
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def reset_env():
    rospy.wait_for_service('/gazebo/reset_world')
    try:
        reset_world_handle = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        return reset_world_handle()
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


def delay():
	i = 1000000
	while i > 0:
		i -= 1


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    rate = rospy.Rate(200) # 10hz

    reset_env()

    while not rospy.is_shutdown():
        for i in range(100):
            udp.mav.rc_channels_override_send(udp.target_system, udp.target_component, 150, 150, 150, 150, 0, 0, 0, 0)
            data = udp.recv_msg()
            next_step()

        print 'stop'

        for i in range(100):
            udp.mav.rc_channels_override_send(udp.target_system, udp.target_component, 0, 0, 0, 0, 0, 0, 0, 0)
            data = udp.recv_msg()
            next_step()

        print 'run'

        for i in range(100):
            udp.mav.rc_channels_override_send(udp.target_system, udp.target_component, 150, 150, 150, 150, 0, 0, 0, 0)
            data = udp.recv_msg()
            next_step()

        print 'reset world'
        reset_env()

if __name__ == '__main__':
    print "Connecting"
    udp = mavutil.mavudp('127.0.0.1:14560', source_system=0)  #gazebo
    listener()