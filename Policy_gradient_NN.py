#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:26:42 2019

@author: siddharth
"""

import gym

import numpy as np
import tensorflow as tf


def initialize_var(no_inputs,neurons_per_hidderlayer_li):
    
    wt_initializer = tf.contrib.layers.variance_scaling_initializer()
    parameters = {}
    
    with tf.variable_scope('Weights'):
        for idx,neurons in enumerate(neurons_per_hidderlayer_li,1):
            if idx == 1:
                parameters["w"+str(idx)] = tf.Variable(wt_initializer([no_inputs,neurons]),dtype = tf.float32,name = 'w'+str(idx))
            else:
                parameters["w"+str(idx)] = tf.Variable(wt_initializer([neurons_per_hidderlayer_li[idx-2],neurons]),dtype = tf.float32,name = 'w'+str(idx))
        
        parameters["w"+str(idx+1)] = tf.Variable(wt_initializer([neurons_per_hidderlayer_li[idx-1],1]),dtype = tf.float32,name = 'w'+str(idx+1))
    
    with tf.variable_scope('Bias'):
        for idx,neurons in enumerate(neurons_per_hidderlayer_li,1):
            if idx == 1:
                parameters["b"+str(idx)] = tf.Variable(np.zeros([no_inputs,neurons]),dtype = tf.float32,name = 'b'+str(idx))
            else:
                parameters["b"+str(idx)] = tf.Variable(np.zeros([neurons_per_hidderlayer_li[idx-2],neurons]),dtype = tf.float32,name = 'b'+str(idx))
        
        parameters["b"+str(idx+1)] = tf.Variable(np.zeros([neurons_per_hidderlayer_li[idx-1],1]),dtype = tf.float32,name = 'b'+str(idx+1))
      
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

def cost_optimizer(y,logits,lr):
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    
    return cost,optimizer
    
def neural_net_policy(no_inputs,neurons_per_hiddenlayer_li,lr):
    
    
    # defining placeholders
    x_ph = tf.placeholder(tf.float32,shape=[None,no_inputs],name = 'x_ph')
    
    # variables initialization
    parameters = initialize_var(no_inputs,neurons_per_hiddenlayer_li)
    
    # defining forward propagation
    logits = forward_prop(x_ph,parameters)
    
    # defining actions and y_labels
    output = tf.sigmoid(logits)    
    p_left_right = tf.concat([output,1-output],axis=1)
    action = tf.multinomial(tf.log(p_left_right),num_samples=1) 
    
        #if we have output close to 1 => p_left is higher hence action will be 0(index) we reverse this in below code
    y = 1-tf.to_float(action) # we use (1-actions) because output measures the prob of left action
    
    # defining cost and optimizer
    cost,optimizer = cost_optimizer(y,logits,lr)
    
    # defining gradients
    grads_vars = optimizer.compute_gradients(cost)    
    gradients = [grad for grad,var in grads_vars]
    
    # creating gradients placeholder
    grads_vars_feed = []
    gradient_placeholders = []
    for grad,var in grads_vars:
        grad_placeholder = tf.placeholder(tf.float32,shape=grad.get_shape())
        gradient_placeholders.append(grad_placeholder)
        grads_vars_feed.append((grad_placeholder,var))
        
    training_op = optimizer.apply_gradients(grads_vars_feed)
    
    return action,gradients,gradient_placeholders,training_op


def discounted_rewards(epi_rewards,discount_rate):
    dis_epi_reward = []
    cumul_reward = 0 
    
    for reward in reversed(epi_rewards):
        cumul_reward = reward + cumul_reward *  discount_rate
        dis_epi_reward.append(cumul_reward)
       
    return reversed(dis_epi_reward)
        
def normalize_rewards(all_rewards,discount_rate):
    
    all_dis_rewards = []
    for epi_rewards in all_rewards:
        dis_rewards = discounted_rewards(epi_rewards,discount_rate)
        all_dis_rewards.append(dis_rewards)
    
    flat_all_rewards = np.concatenate(all_dis_rewards)
    mean = np.mean(flat_all_rewards)
    std = np.std(flat_all_rewards)
    
    return [ (rewards-mean)/std for rewards in all_dis_rewards]
    

if __name__ == '__main__':
    
    # creating environment
    env = gym.make('CartPole-v0')
    
    
    # defining neural network
    learning_rate = 0.01
    no_inputs = len(env.reset())
    neurons_per_hiddenlayer_li = [4]
    
    action,gradients,gradient_placeholders,training_op = neural_net_policy(no_inputs,neurons_per_hiddenlayer_li,learning_rate)
    
    # defining cartpole game parameters
    no_training_epochs = 250
    update_after_episode = 10
    max_steps_per_episode = 1000
    
    discount_rate = 0.95
    
    with tf.Session() as sess:
        for epoch in range(no_training_epochs):
            all_rewards = []
            all_gradients = []
            
            for ep in range(update_after_episode):
                current_rewards = []
                current_gradients = []
                obs = env.reset()
                
                for step in range(max_steps_per_episode):                
                    ac,gr = sess.run([action,gradients],feed_dict=obs.reshape((1,len(obs))))                    
                    obs,reward,done,info = env.step(ac)
                    
                    current_rewards.append(reward)
                    current_gradients.append(gr)
                    
                    if done:
                        break
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)
             
            
            all_rewards = normalize_rewards(all_rewards,discount_rate)
            
            training_feed = {}
            for grad_idx,grad_placeholder in enumerate(gradient_placeholders):
                
                mean_grad = np.mean(
                        [ all_rewards[epi_idx][step]* all_gradients[epi_idx][step][grad_idx] 
                         for epi_idx,epi_rewards in enumerate(all_rewards)
                            for step,reward in range(epi_rewards)]
                        ,axis=0)
                training_feed[grad_placeholder] = mean_grad
            
            # training NN with modified gradients                
            sess.run(training_op,feed_dict=training_feed)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        