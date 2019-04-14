#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:46:32 2019

@author: siddharth
"""

import gym
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