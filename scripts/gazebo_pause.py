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


def delay():
	i = 1000000
	while i > 0:
		i -= 1

def arm_quad():
    rospy.wait_for_service('mavros/cmd/arming')
    try:
        arm_handle = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
        resp = arm_handle(True)
        print resp.success
        print('quad arming success:',resp.success)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def set_offboard_mode():
    rospy.wait_for_service('mavros/set_mode')
    try:
        set_mode_handle = rospy.ServiceProxy('mavros/set_mode', SetMode)
        set_mode_handle(0, "OFFBOARD")
        print 'Set mode to OFFBOARD'
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def px4_node_restart():
    launch.shutdown()
    launch.start()



def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/indigo/src/Firmware/launch/px4_posix_sitl.launch"])

    launch.start()


    pub = rospy.Publisher('/mavros/actuator_control', ActuatorControl, queue_size=10)
    rate = rospy.Rate(200) # 10hz
    set_offboard_mode()
    # next_step()
    # next_step()
    delay()
    arm_quad()
    # next_step()
    # next_step()

    while not rospy.is_shutdown():
        # delay()
        actuatorControl = ActuatorControl()
        actuatorControl.group_mix = actuatorControl.PX4_MIX_MANUAL_PASSTHROUGH
        actuatorControl.controls = [0, 0, 0, 1, 0, 0, 0, 0]
        pub.publish(actuatorControl)
        # next_step()
        rate.sleep()

if __name__ == '__main__':
    listener()