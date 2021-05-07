#!/bin/bash

source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI="http://10.42.0.1:11311"
export ROS_IP=`ip -o -4 addr list | awk '$4~/10\.42\.0\./ {print $4}' | cut -d/ -f1`
