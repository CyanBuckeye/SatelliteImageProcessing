# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 22:54:44 2016

@author: Administrator

define Caffe network
"""

import sys
import caffe
import os

from caffe import layers as L
from caffe import params as P
file_path = r''
train_data = r'train.h5list'
val_data = r'val.h5list'

def Network(hdfData, batch_size):
    net = caffe.NetSpec()
    net.data, net.label = L.HDF5Data(batch_size = batch_size, source = hdfData, ntop = 2)
    net.conv1 = L.Convolution(net.data, kernel_size = 2, stride = 2, num_output = 128, weight_filler = dict(type = 'gaussian', std = 0.01))
    net.relu1 = L.ReLU(net.conv1, in_place = True)
    net.pool1 = L.Pooling(net.relu1, kernel_size = 2, stride = 1, pad = 1, pool = P.Pooling.MAX)
    net.conv2 = L.Convolution(net.pool1, kernel_size = 2, stride = 2, num_output = 512, weight_filler  = dict(type = 'gaussian', std = 0.01))
    net.relu2 = L.ReLU(net.conv2, in_place = True)
    net.pool2 = L.Pooling(net.relu2, kernel_size = 2, stride = 1, pad = 1, pool = P.Pooling.MAX)
    net.conv3 = L.Convolution(net.pool2, kernel_size = 2, stride = 2, num_output = 512, weight_filler  = dict(type = 'gaussian', std = 0.01))
    net.relu3 = L.ReLU(net.conv3, in_place = True)
    net.pool3 = L.Pooling(net.relu3, kernel_size = 2, stride = 1, pad = 1, pool = P.Pooling.MAX)
    net.conv4 = L.Convolution(net.pool3, kernel_size = 1, stride = 1, num_output = 2048, weight_filler  = dict(type = 'gaussian', std = 0.01))
    net.conv5 = L.Convolution(net.conv4, kernel_size = 1, stride = 1, num_output = 2048, weight_filler  = dict(type = 'gaussian', std = 0.01))
    net.conv6 = L.Convolution(net.conv5, kernel_size = 1, stride = 1, num_output = 6, weight_filler  = dict(type = 'gaussian', std = 0.01))
    net.fc0 = L.InnerProduct(net.data, num_output = 256, weight_filler = dict(type = 'xavier'))
    net.relu2 = L.ReLU(net.fc0, in_place = True)
    return net.to_proto()

with open(os.path.join(file_path, 'net1_train.prototxt'), 'w') as f:
    f.write(str(Network(os.path.join(file_path, train_data), 50)))

with open(os.path.join(file_path, 'net1_val.prototxt'), 'w') as f:
    f.write(str(Network(os.path.join(file_path, val_data), 50)))
    
