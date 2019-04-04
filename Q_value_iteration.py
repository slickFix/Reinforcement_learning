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
possible_actions = [[0,1,2],[0,2][1]]

