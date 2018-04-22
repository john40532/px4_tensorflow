#!/bin/bash

sudo apt install -y cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
sudo apt install python3-numpy python3-dev
sudo -H pip3 install gym[all]
sudo -H pip3 install box2d box2d-kengz