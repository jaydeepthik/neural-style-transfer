# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:40:37 2018

@author: jaydeep thik
"""

import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import scipy
#from PIL import Image
from nst_utils import save_image, reshape_and_normalize_image, load_vgg_model, generate_noise_image




STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    with tf.device('/gpu:0'):
        a_C_unrolled = tf.reshape(a_C, [n_C, n_H*n_W])
        a_G_unrolled = tf.reshape(a_G, [n_C, n_H*n_W])
        
        J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))*(1./(2 * n_H**0.5 * n_W**0.5 * n_C**0.5))
    return J_content


def gram_matrix(A):
    with tf.device('/gpu:0'):
        return tf.matmul(A, tf.transpose(A))


def layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    with tf.device('/gpu:0'):
        a_S = tf.transpose(tf.reshape(a_S, [-1,n_C]))
        a_G = tf.transpose(tf.reshape(a_G, [-1,n_C]))
        
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        
        J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS,GG)))*(1./(4*n_H**2 * n_W**2 * n_C**2))
        return np.sum(J_style_layer)


def style_cost(model, STYLE_LAYERS):
    J_style = 0
    for layer_name, lm in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = layer_style_cost(a_S, a_G)
        J_style += lm * J_style_layer
    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 100):
    J = alpha*J_content + beta*J_style
    return J



################################################################################
model = load_vgg_model('./model/imagenet-vgg-verydeep-19.mat')
with tf.device('/gpu:0'):
    sess = tf.InteractiveSession()
    style_image =scipy.misc.imread( 'images/prisma3.jpg')
    
    style_image= scipy.misc.imresize(style_image,(300,400, 3))
    #print(style_image.shape)
    style_image = reshape_and_normalize_image(style_image)
    
    content_image = scipy.misc.imread('images/love.jpeg')
    content_image= scipy.misc.imresize(content_image,(300,400, 3))
    #plt.imshow(content_image)
    content_image = reshape_and_normalize_image(content_image)
    
    generated_image = generate_noise_image(content_image)
    
    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    a_C = sess.run(out)
    a_G = out
    J_content = content_cost(a_C, a_G)
    
    
    sess.run(model['input'].assign(style_image))
    J_style = style_cost(model, STYLE_LAYERS)
    
    J = total_cost(J_content, J_style, 1, 2e2)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1.2)
    train_step = optimizer.minimize(J)

def nn_model(sess, input_image, num_iterations=300):
    with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(input_image))
        
        for i in range(num_iterations):
            sess.run(train_step)
            generated_image = sess.run(model['input'])
            
            if i%20 == 0:
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))
                
                
                save_image("output/test_gpu0/a_" + str(i) + ".png", generated_image)
        
        
        save_image('output/test_gpu0/generated_image.jpg', generated_image)
        
        return generated_image

nn_model(sess, generated_image)

