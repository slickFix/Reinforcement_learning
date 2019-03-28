#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:26:42 2019

@author: siddharth
"""

import gym

import numpy as np
import tensorflow as tf


def initialize_var(no_inputs,neurons_per_layer_li):
    
    wt_initializer = tf.contrib.layers.variance_scaling_initializer()
    parameters = {}
    
    with tf.variable_scope('Weights'):
        for idx,neurons in enumerate(neurons_per_layer_li,1):
            if idx == 1:
                parameters["w"+str(idx)] = tf.Variable(wt_initializer([no_inputs,neurons]),dtype = tf.float32,name = 'w'+str(idx))
            else:
                parameters["w"+str(idx)] = tf.Variable(wt_initializer([neurons_per_layer_li[idx-2],neurons]),dtype = tf.float32,name = 'w'+str(idx))
        
        parameters["w"+str(idx+1)] = tf.Variable(wt_initializer([neurons_per_layer_li[idx-1],1]),dtype = tf.float32,name = 'w'+str(idx+1))
    
    with tf.variable_scope('Bias'):
        for idx,neurons in enumerate(neurons_per_layer_li,1):
            if idx == 1:
                parameters["b"+str(idx)] = tf.Variable(np.zeros([no_inputs,neurons]),dtype = tf.float32,name = 'b'+str(idx))
            else:
                parameters["b"+str(idx)] = tf.Variable(np.zeros([neurons_per_layer_li[idx-2],neurons]),dtype = tf.float32,name = 'b'+str(idx))
        
        parameters["b"+str(idx+1)] = tf.Variable(np.zeros([neurons_per_layer_li[idx-1],1]),dtype = tf.float32,name = 'b'+str(idx+1))
      
    return parameters
        
    

def forward_prop(x_ph,parameters):
    
    weights = [ val for val in sorted(parameters.keys()) if 'w' in val ]
    bias = [ val for val in sorted(parameters.keys()) if 'b' in val ]
    
    
    layers = {}
    
    with tf.variable_scope('Layers'):
        for idx,(w,b) in enumerate(zip(weights,bias),1):
            if idx == 1:
                layers['layer'+str(idx)] = tf.nn.elu(tf.add(tf.matmul(x_ph,w),b))
            elif idx!=len(weights):
                layers['layer'+str(idx)] = tf.nn.elu(tf.add(tf.matmul(layers['layer'+str(idx-1)],w),b))
        logits = tf.add(tf.matmul(layers['layer'+str(idx-1)],weights[-1]),bias[-1])
    
    return logits

def neural_net_policy(no_inputs):
    
    
    # defining placeholders
    x_ph = tf.placeholder(tf.float32,shape=[None,no_inputs],name = 'x_ph')
    
    # variables initialization
    parameters = initialize_var(no_inputs,neurons_per_layer_li)
    
    # defining forward propagation
    logits = forward_prop(x_ph,parameters)
    
     

if __name__ == '__main__':
    
    # creating environment
    env = gym.make('CartPole-v0')
    
    
    # defining neural network
    
    

        
        