#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:25:05 2017

@author: xu.2727
test well-trained model
"""

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import pickle
import numpy as np
import os
import scipy.io as sio
from PIL import Image
import pdb

channel = 6
imgSize = 128

image_path = ''#path to image
db_root = ''#path to data
filePath = '1_indcell.mat'#dsm feature of test image
variable_name = 'newCell'

class_map = {
0:(255,255,255),
1:(0,0,255),
2:(0,255,255),
3:(0,255,0),
4:(255,255,0)       
}

color_map = {
(255,255,255):0,
(0,0,255):1,
(0,255,255):2,
(0,255,0):3,
(255,255,0):4,
(255,0,0):5          
}

class Net(nn.Module):
    def __init__(self, inputSize):
        super(Net, self).__init__()
        channel, width, height = inputSize
                                 
        self.block1 = nn.Sequential(OrderedDict([
                ('conv1_1', nn.Conv2d(channel, 32, kernel_size=3, stride=1)),
                ('BN1_1', nn.BatchNorm2d(32)),
                ('relu1_1', nn.PReLU(32)),
                ('conv1_2', nn.Conv2d(32, 32, kernel_size=1, stride=1)),
                ('BN1_2', nn.BatchNorm2d(32)),
                ('relu1_2', nn.PReLU(32)),
                ('conv1_3', nn.Conv2d(32, 32, kernel_size=1, stride=1)),
                ('BN1_3', nn.BatchNorm2d(32)),
                ('relu1_3', nn.PReLU(32)),
                ]))                         

        self.block2 = nn.Sequential(OrderedDict([
                ('conv2_1', nn.Conv2d(32, 64, kernel_size=3, stride=1)),
                ('BN2_1', nn.BatchNorm2d(64)),
                ('relu2_1', nn.PReLU(64)),
                ('conv2_2', nn.Conv2d(64, 64, kernel_size=1, stride=1)),
                ('BN2_2', nn.BatchNorm2d(64)),
                ('relu2_2', nn.PReLU(64)),
                ('conv2_3', nn.Conv2d(64, 64, kernel_size=1, stride=1)),
                ('BN2_3', nn.BatchNorm2d(64)),
                ('relu2_3', nn.PReLU(64)),
                ]))  
        
        self.block3 = nn.Sequential(OrderedDict([
                ('conv3_1', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
                ('BN3_1', nn.BatchNorm2d(128)),
                ('relu3_1', nn.PReLU(128)),
                ('conv3_2', nn.Conv2d(128, 128, kernel_size=1, stride=1)),
                ('BN3_2', nn.BatchNorm2d(128)),
                ('relu3_2', nn.PReLU(128)),
                ('conv3_3', nn.Conv2d(128, 128, kernel_size=1, stride=1)),
                ('BN3_3', nn.BatchNorm2d(128)),
                ('relu3_3', nn.PReLU(128)),
                ('conv3_4', nn.Conv2d(128, 128, kernel_size=3, stride=1)),
                ('BN3_4', nn.BatchNorm2d(128)),
                ('relu3_4', nn.PReLU(128))
                ]))  
        
        self.block4 = nn.Sequential(OrderedDict([
                ('conv4_1', nn.Conv2d(128, 256, kernel_size=2, stride=1)),
                ('BN4_1', nn.BatchNorm2d(256)),
                ('relu4_1', nn.PReLU(256)),
                ('conv4_2', nn.Conv2d(256, 256, kernel_size=1, stride=1)),
                ('BN4_2', nn.BatchNorm2d(256)),
                ('relu4_2', nn.PReLU(256)),
                ('conv4_3', nn.Conv2d(256, 256, kernel_size=1, stride=1)),
                ('BN4_3', nn.BatchNorm2d(256)),
                ('relu4_3', nn.PReLU(256)),
                ('conv4_4', nn.Conv2d(256, 256, kernel_size=1, stride=1)),
                ('BN4_4', nn.BatchNorm2d(256)),
                ('relu4_4', nn.PReLU(256))
                ]))          
        
        self.block5 = nn.Sequential(OrderedDict([
                ('deconv5_1', nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=1, padding=0, output_padding=0)),
                ('BN5_1', nn.BatchNorm2d(128)),
                ('relu5_1', nn.PReLU(128)),
                ('upSample5_1', nn.UpsamplingBilinear2d(scale_factor=2))
        ]))
        self.block6 = nn.Sequential(OrderedDict([
                ('deconv6_1', nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding=0, output_padding=0)),
                ('BN6_1', nn.BatchNorm2d(64)),
                ('relu6_1', nn.PReLU(64)),
                ('upSample6_1', nn.UpsamplingBilinear2d(scale_factor=2))
        ]))
        self.block7 = nn.Sequential(OrderedDict([
                ('deconv7_1', nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=1, padding=0, output_padding=0)),
                ('BN7_1', nn.BatchNorm2d(32)),
                ('relu7_1', nn.PReLU(32)),
                ('upSample7_1', nn.UpsamplingBilinear2d(scale_factor=2))
        ]))      
        self.block8 = nn.Sequential(OrderedDict([
                ('deconv8_1', nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=1, padding=0, output_padding=0)),
                ('BN8_1', nn.BatchNorm2d(16)),
                ('relu8_1', nn.PReLU(16)),
                ('upSample8_1', nn.UpsamplingBilinear2d(scale_factor=2)),
                ('deconv8_2', nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0, output_padding=0))
        ]))      
        self.block9 = nn.Sequential(OrderedDict([
                ('conv9_1', nn.Conv2d(16, 32, kernel_size=1, stride=1)),
                ('BN9_1', nn.BatchNorm2d(32)),
                ('relu9_1', nn.PReLU(32)),
                ('conv9_2', nn.Conv2d(32, 32, kernel_size=1, stride=1)),
                ('BN9_2', nn.BatchNorm2d(32)),
                ('relu9_2', nn.PReLU(32)),
                ('score', nn.Conv2d(32, 5, kernel_size=1, stride=1))
                ]))

    def forward(self, x):
        blk1 = self.block1(x)
        pool1 = F.max_pool2d(blk1, kernel_size=2, stride=2)
        
        
        blk2 = self.block2(pool1)
        pool2 = F.max_pool2d(blk2, kernel_size=2, stride=2)
        
        blk3 = self.block3(pool2)
        pool3 = F.max_pool2d(blk3, kernel_size=2, stride=2)
        
        blk4 = self.block4(pool3)
        pool4 = F.max_pool2d(blk4, kernel_size=2, stride=2)
        
        blk5 = self.block5(pool4)
        
        blk6 = self.block6(blk5)
        
        blk7 = self.block7(blk6)
        blk8 = self.block8(blk7)
        
        prob = self.block9(blk8)
        prob = F.log_softmax(prob)
  
        return prob

net = Net([channel,imgSize,imgSize])
net.cuda()
net.load_state_dict(torch.load('./net.pth'))
net.eval()

origin_name = 'top_mosaic_09cm_area' + str(1) + '.tif'
origin = Image.open(os.path.join(image_path, origin_name))# get Data
origin = np.asarray(origin)
dsm_name = str(1) + '_data.mat'
dsm_file = sio.loadmat(os.path.join(dsm_root, dsm_name))
dsm = dsm_file['result']#get dsm
dsm = dsm * 255
dsm = dsm.astype(np.uint8)

result = np.zeros(origin.shape, dtype=np.uint8)
img_height = origin.shape[0]
img_width = origin.shape[1]

for i in range(imgSize, img_height, imgSize):
    for j in range(imgSize, img_width, imgSize):
        print(i * img_width + j)
        inputData = np.zeros((1, channel, imgSize, imgSize), dtype=np.float32)
        inputData[0][:3] = origin[i - imgSize : i, j - imgSize : j].transpose(2,0,1)
        inputData[0][3:6] = dsm[i - imgSize : i, j - imgSize : j].transpose(2,0,1)
        inputVariable = Variable(torch.from_numpy(inputData)).cuda()
        
        output = net(inputVariable)
        output = output.cpu().data.numpy()[0]
        output = output.transpose(1,2,0)
        
        tempResult = np.zeros((imgSize, imgSize, 3), dtype=np.uint8)
        for k in range(output.shape[0]):
            for l in range(output.shape[1]):
                tempResult[k][l] = class_map[output[k][l].argmax()]
        result[i - imgSize : i, j - imgSize : j] = tempResult

imgLabel = sio.loadmat(filePath)
indcell = imgLabel[variable_name]

output = np.zeros(origin.shape, dtype=np.uint8)
for i in range(indcell.shape[0]):
    print('%d / %d' % (i, indcell.shape[0]))
    size = len(indcell[i][0][0])
    count = np.zeros((5,), dtype=np.int32)
    for j in range(size):
        x = indcell[i][0][0][j] - 1
        y = indcell[i][1][0][j] - 1
        temp_color = tuple(result[x][y])
        if temp_color not in color_map.keys():
            continue
        label = color_map[temp_color]
        count[label] += 1
    idx = count.argmax()
    color = class_map[idx]
    for j in range(size):
        x = indcell[i][0][0][j] - 1
        y = indcell[i][1][0][j] - 1
        output[x][y] = color
resultImg = Image.fromarray(output)
resultImg.save('predict.jpg')