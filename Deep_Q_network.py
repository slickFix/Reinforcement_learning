#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:46:32 2019

@author: siddharth
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d,fully_connected


env = gym.make("MsPacman-v0")
obs = env.reset()
obs.shape
env.action_space

mspacman_color = np.array([210,164,74]).mean()

def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.mean(axis = 2) # to greyscale
    img[img  == mspacman_color] = 0 # improve contrast
    img = (img-128)/128-1 # normalize from -1 to 1
    return img.reshape(88,80,1)

input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32,64,64]
conv_kernel_sizes = [(8,8),(4,4),(3,3)]
conv_strides = [4,2,1]
conv_paddings = ["SAME"]*3
conv_activation = [tf.nn.relu]*3
n_hidden_in = 64*11*10 # conv3  has 64 maps of 11*10 each
n_hidden = 512

hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # 9 discrete actions are available
initializer = tf.contrib.layers.variance_scaling_initializer()
