#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net: Convolutional Networks for Biomedical Image Segmentation  
Olaf Ronneberger, Philipp Fischer, Thomas Brox
Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, 
LNCS, Vol.9351: 234--241, 2015, available at arXiv:1505.04597 [cs.CV] 


Created on Thu Jul 26 10:20:18 2018
@author: onur
"""
import tensorflow as tf
import sys
import numpy as np
import tflearn
import glob
from tflearn.layers.normalization import batch_normalization as bn
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d


class Unet:
    def __init__(self, min_filter = 32, max_filter = 256,classnumber=2, name = "g",padding = 'SAME'):
        """
        Initialize a UNet generator.
        :param min_filter: Number of  conv filters ( beginning and ending of U s)
        :param max_filter: Number of filters (Deep of U shape, middle layer)
        :param classnumber: Number of classes
        """
        self.min_filter = min_filter
        self.max_filter = max_filter
        self.name = name
        self.padding = padding
        self.classnumber = classnumber
    def bn(self,inputs):
        return tf.layers.batch_normalization(inputs)

    def conv_layer(self, inputs, num_filters, filter_size=3, strides=1, activation=None,
                   padding="SAME",kernel_init= tf.uniform_unit_scaling_initializer()):
        """
        Add a convolutional layer to the network.
        :param inputs: Inputs to the conv layer.
        :param num_filters: Num of filters for conv layer.
        :param filter_size: Size of filter.
        :param strides: Stride size.
        :param activation: Conv layer activation.
        :return: Convolution features
        """     

        outputs = tf.layers.conv2d(inputs, num_filters, filter_size, strides=strides,
                                                 padding=padding, activation=activation,
                                                 kernel_initializer= kernel_init,
                                                 bias_initializer=tf.zeros_initializer())

        return outputs
    def unet_layer(self,inputs,num_filters):
        output = self.conv_layer(inputs,num_filters,kernel_init=tf.glorot_normal_initializer())
        output = self.bn(tf.nn.relu(output))
        output = self.conv_layer(inputs,num_filters,kernel_init=tf.glorot_normal_initializer())
        output = self.bn(tf.nn.relu(output))
        return output
       
    def __call__(self,X,name,reuse):
        itr=self.min_filter
        adds=[]
        indx=0
        with tf.variable_scope(name,reuse=reuse):
             while(itr<=self.max_filter):
                 if(itr== self.min_filter):
                     output=self.unet_layer(X,itr)
                 else:
                     output=self.unet_layer(output,itr)
                 if (itr<self.max_filter):
                     adds.append(output)
#                 output=tf.layers.max_pooling2d(output,[2,2],1)
                 if(itr<self.max_filter):
                     output=max_pool_2d(output,2)
                 itr=itr*2
                 #print(itr)
             itr=itr/2
             #print(output.get_shape().as_list())
             while(itr>self.min_filter):
                 itr=itr/2
                 #print(itr)
                 indx=indx-1
                 #print(indx)
                 shape = output.get_shape().as_list()
                 output = tf.image.resize_nearest_neighbor(output, (shape[1]*2, shape[2]*2))
                 output = self.unet_layer(output,itr)
                 temp = tf.image.resize_nearest_neighbor(adds[indx], (shape[1]*2, shape[2]*2))
                 output=tf.concat([output,temp],3)
             output=self.conv_layer(output,self.classnumber,activation=tf.nn.relu)
             return output
            


