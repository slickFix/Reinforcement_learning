#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:21:57 2019

@author: siddharth
"""

import numpy as np


# Defining Markov Decision Process

# represents impossible actions
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

# possible actions
possible_actions = [[0,1,2],[0,2],[1]]

# Running Q Value iteration alogrithm

Q = np.full((3,3),-np.inf)

for state,actions in enumerate(possible_actions):
    Q[state,actions] = 0.0
print(Q)    
learning_rate = 0.01
discount_rate = 0.95
n_iterations = 100

for iteration in range(n_iterations):
    Q_prev = Q.copy()
    for s in range(3):
        for a in possible_actions[s]:            
            Q[s,a] = np.sum([
                    T[s,a,sp]*(R[s,a,sp]+discount_rate*np.max(Q_prev[sp])) for sp in range(3)
                    ])
    
print(Q)