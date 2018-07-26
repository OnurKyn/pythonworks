#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 10:20:18 2018

@author: onur
"""
import tensorflow as tf
import cv2
import sys
import numpy as np
import tflearn
import glob
from tflearn.layers.normalization import batch_normalization as bn
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.core import fully_connected as fc
from mylib.mycv import goster,Deconv
class Unet:
    def __init__(self, layer_sizes, layer_padding, batch_size, num_channels=1,
                 inner_layers=0, name="g"):
        """
        Initialize a UNet generator.
        :param layer_sizes: A list with the filter sizes for each MultiLayer e.g. [64, 64, 128, 128]
        :param layer_padding: A list with the padding type for each layer e.g. ["SAME", "SAME", "SAME", "SAME"]
        :param batch_size: An integer indicating the batch size
        :param num_channels: An integer indicating the number of input channels
        :param inner_layers: An integer indicating the number of inner layers per MultiLayer
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_sizes = layer_sizes
        self.layer_padding = layer_padding
        self.inner_layers = inner_layers
        self.conv_layer_num = 0
        self.build = True
        self.name = name
    def bn(self,inputs):
        return tf.layers.batch_normalization(inputs)

    def conv_layer(self, inputs, num_filters, filter_size, strides, activation=None,
                   padding="SAME",reluabn=True):
        """
        Add a convolutional layer to the network.
        :param inputs: Inputs to the conv layer.
        :param num_filters: Num of filters for conv layer.
        :param filter_size: Size of filter.
        :param strides: Stride size.
        :param activation: Conv layer activation.
        :param transpose: Whether to apply upscale before convolution.
        :param w_size: Used only for upscale, w_size to scale to.
        :param h_size: Used only for upscale, h_size to scale to.
        :return: Convolution features
        """      

        outputs = tf.layers.conv2d(inputs, num_filters, filter_size, strides=strides,
                                                 padding=padding, activation=activation)
        if reluabn:
            outputs = self.bn(inputs=outputs)
            outputs = tf.nn.relu(outputs)
        return outputs
    def Generator(self,X,name,reuse):
         with tf.variable_scope(name,reuse=reuse):
             return self.conv_layer(X,32,3,1)
             
             
            #1
#            conv1_1 = conv_2d(X, 32, 3, 1,padding="valid", activation='relu', name="conv1_1")#254
#            conv1_1 = bn(conv1_1)
#            conv1_2 = conv_2d(conv1_1, 32, 3, 1,padding="valid", activation='relu', name="conv1_2")#252
#            conv1_2 = bn(conv1_2)
#            pool1 = max_pool_2d(conv1_2, 2)#126
#            
#            
#            #2
#            conv2_1 = conv_2d(pool1, 64, 3, 1,padding="valid", activation='relu', name="conv2_1")#124
#            conv2_1 = bn(conv2_1)
#            conv2_2 = conv_2d(conv2_1, 64, 3, 1,padding="valid", activation='relu', name="conv2_2")#122
#            conv2_2 = bn(conv2_2)
#            pool2 = max_pool_2d(conv2_2, 2)#61
#            
#            
#            #3
#            conv3_1 = conv_2d(pool2, 128, 3, 1, padding="valid",activation='relu', name="conv3_1")#59
#            conv3_1 = bn(conv3_1)
#            conv3_2 = conv_2d(conv3_1, 128, 3, 1, padding="valid",activation='relu', name="conv3_2")#57
#            conv3_2 = bn(conv3_2)
#            pool3 = max_pool_2d(conv3_2, 2)#29
#                   
#            
#            conv4_1 = conv_2d(pool3, 256, 3, 1,padding="valid", activation='relu', name="conv4_1")#27
#            conv4_1 = bn(conv4_1)
#            conv4_2 = conv_2d(conv4_1, 256, 3, 1,padding="valid", activation='relu', name="conv4_2")#25
#            conv4_2 = bn(conv4_2)
#            pool4 = max_pool_2d(conv4_2, 2)#13
#            
#            conv5_1 = conv_2d(pool4, 512, 3, 1,padding="valid", activation='relu', name="conv5_1")#11
#            conv5_1 = bn(conv5_1)
#            conv5_2 = conv_2d(conv5_1, 512, 3, 1,padding="valid", activation='relu', name="conv5_2")#9
#            conv5_2 = bn(conv5_2)
#            conv5_2_upsampled=upsample_2d(conv5_2, 2)
#            
#            shape=conv5_2_upsampled.get_shape().as_list()
#            up6 = merge([conv5_2_upsampled, tf.image.resize_images(conv4_2,[shape[1],shape[2]])], mode='concat', axis=3, name='upsamle-5-merge-4')
#            conv6_1 = conv_2d(up6, 256, 3, 1,padding="valid", activation='relu', name="conv6_1")
#            conv6_1 = bn(conv6_1)
#            conv6_2 = conv_2d(conv6_1, 256, 3, 1,padding="valid", activation='relu', name="conv6_2")
#            conv6_2 = bn(conv6_2)
#            conv6_2_upsampled=upsample_2d(conv6_2, 2)
#            
#            shape=conv6_2_upsampled.get_shape().as_list()
#            up7 = merge([conv6_2_upsampled, tf.image.resize_images(conv3_2,[shape[1],shape[2]])], mode='concat', axis=3, name='upsamle-6-merge-3')
#            conv7_1 = conv_2d(up7, 128, 3, 1, padding="valid",activation='relu', name="conv7_1")
#            conv7_1 = bn(conv7_1)
#            conv7_2 = conv_2d(conv7_1, 128, 3, 1, padding="valid",activation='relu', name="conv7_2")
#            conv7_2 = bn(conv7_2)
#            conv7_2_upsampled=upsample_2d(conv7_2, 2)
#            
#            shape=conv7_2_upsampled.get_shape().as_list()
#            up8 = merge([conv7_2_upsampled, tf.image.resize_images(conv2_2,[shape[1],shape[2]])], mode='concat', axis=3, name='upsamle-7-merge-2')        
#            conv8_1 = conv_2d(up8, 64, 3, 1,padding="valid", activation='relu', name="conv8_1")
#            conv8_1 = bn(conv8_1)
#            conv8_2 = conv_2d(conv8_1, 64, 3, 1,padding="valid", activation='relu', name="conv8_2")
#            conv8_2 = bn(conv8_2)
#            conv8_2_upsampled=upsample_2d(conv8_2, 2)
#            
#            shape=conv8_2_upsampled.get_shape().as_list()
#            up9 = merge([conv8_2_upsampled, tf.image.resize_images(conv1_2,[shape[1],shape[2]])], mode='concat', axis=3, name='upsamle-8-merge-1')
#            conv9_1 = conv_2d(up9, 32, 3, 1,padding="valid", activation='relu', name="conv9_1")
#            conv9_1 = bn(conv9_1)
#            conv9_2 = conv_2d(conv9_1, 32, 3, 1,padding="valid", activation='relu', name="conv9_2")
#            conv9_2 = bn(conv9_2)
#            
#            return conv_2d(conv9_2, 2, 1, 1,padding="valid", activation='linear', name="conv10")
