#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:47:33 2019

@author: siddharth
"""


import gym
import numpy as np


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle<0 else 1


if __name__ == '__main__':
    
    env = gym.make('CartPole-v0')  # defining training environment    
    obs = env.reset()  # initializing env
    
    print("initial observations : ",obs)
    
    env.render()  # showing enviroment gui
    env.action_space  # possible actions in case 0(-ve acceleration) or 1(+ve acceleration)
    
    action = 0
    obs,reward,done,info = env.step(action)  # returns obs,reward,done and info after the step is performed
    
    print('obs ',obs)
    print('reward ',reward)
    print('done ',done)
    print('info ',info)
    
    totals = []
    
    for episode in range(500):
        episode_rewards = 0
        obs = env.reset()
        for step in range(1000):
            action = basic_policy(obs)
            obs,reward,done,info = env.step(action)
            episode_rewards+=reward
            if done:
                break
        totals.append(episode_rewards)
    
    print("Mean of totals : ",np.mean(totals))
    print("Std of totals : ",np.std(totals))
    print("Min of totals : ",np.min(totals))
    print("Max of totals : ",np.max(totals))
            
