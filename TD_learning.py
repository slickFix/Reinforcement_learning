#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:57:01 2019

@author: siddharth
"""
import numpy as np
import numpy.random as rnd

learning_rate0 = 0.05
learning_rate_decay = 0.1
n_iterations = 20000
discount_rate = 0.95

nan = np.nan

# Transition probability matrix
# shape = [s,a,s`]
T = np.array([
        [[0.7,0.3,0.0],[1.0,0.0,0.0],[0.8,0.2,0.0]],
        [[0.0,1.0,0.0],[nan,nan,nan],[0.0,0.0,1.0]],
        [[nan,nan,nan],[0.8,0.1,0.1],[nan,nan,nan]]
                ])

# Reward matrix
# shape = [s,a,s`]
R = np.array([
        [[10,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[nan,nan,nan],[0,0,-50]],
        [[nan,nan,nan],[40,0,0],[nan,nan,nan]]
           ])

s = 0 # start in state 0

Q = np.full((3,3),-np.inf)  # -inf for impossible actions

# possible actions
possible_actions = [[0,1,2],[0,2],[1]]

for state,actions in enumerate(possible_actions):
    Q[state,actions] = 0.0   # initial value = 0.0 for all possible actions

print("T",T)    
print("Q",Q)
print("T.shape",T.shape)
print("Q.shape",Q.shape)    

print(list(range(3)))

for iteration in range(n_iterations):
    a = rnd.choice(possible_actions[s])  # choose an action(randomly)
    sp = rnd.choice(range(3),p = T[s,a]) # pick next state using T[s,a]
    
    reward = R[s,a,sp]
    
    learning_rate = learning_rate0/(1+iteration * learning_rate_decay)
    
    Q[s,a] = learning_rate *Q[s,a] + (1-learning_rate)*(reward + discount_rate * np.max(Q[sp]))
    
    s = sp

print(Q)