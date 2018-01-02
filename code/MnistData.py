# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 16:43:25 2018
download Minist Data
@author: dell
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, source_url='http://yann.lecun.com/exdb/mnist/')