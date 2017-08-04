#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: xu.2727
deconvolution network for pixel-wise classification
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
import pdb

db_root = ''#path to data
channel = 6 #channel of data RGB + DSM(three-channel height feature)
imgSize = 128 #image size
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

lr = 0.01
img_list = [0, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]#training image
start_idx = 1
end_idx = len(img_list)
show = 20
epoch = 5
miniBatch = 5 #batch size

load_flag = True
for e in range(epoch):
    optimizer = torch.optim.SGD(net.parameters(), lr = lr)
    for img in range(start_idx, end_idx):
        if load_flag == True:
            net.load_state_dict(torch.load('./net.pth'))
        load_flag = True
        data_path = os.path.join(db_root, 'trainData' + str(img_list[img]) + '.p')
        valData_path = os.path.join(db_root, 'valData' + str(img_list[img]) + '.p')
    
        f = open(data_path, 'rb')
        data = pickle.load(f)
        f.close()
        
        f = open(valData_path, 'rb')
        val_data = pickle.load(f)
        f.close()
        
        
        num = data.shape[0]
        batch = num / miniBatch
        loss_accum = 0.0
        
        testNum = val_data.shape[0]
        
        print('finish data loading')
        
        net.eval()
        test_loss = 0
        for i in range(testNum):
            test_data = val_data[i, :6]
            test_data = test_data.astype(np.float32).reshape(1,6,128,128)
            test_data = torch.from_numpy(test_data)
            input_data = Variable(test_data).cuda()
            test_label = val_data[i][6]
            test_label = test_label.reshape(imgSize, imgSize)
            
            output = net(input_data)
            output = output.cpu().data.numpy()[0]
            output = output.transpose(1,2,0)
            
            count = 0
            for h in range(imgSize):
                for w in range(imgSize):
                    if output[h,w].argmax() <> test_label[h][w]:
                        count += 1
            test_loss += count / float(imgSize * imgSize)
        print(test_loss / float(testNum))
        net.train()
        
        np.random.shuffle(data)
        for i in range(batch):
            temp_data = data[i * miniBatch : i * miniBatch + miniBatch, :6]
            temp_data = temp_data.astype(np.float32)
            temp_data = torch.from_numpy(temp_data)
            train_data = Variable(temp_data).cuda()
            temp_label = data[i * miniBatch : i * miniBatch + miniBatch, 6]
            temp_label = temp_label.reshape(miniBatch, imgSize, imgSize)
            temp_label = temp_label.astype(np.int64)
            
            temp_label = torch.from_numpy(temp_label)
            train_label = Variable(temp_label).cuda()
            
            optimizer.zero_grad()
            outputs = net(train_data)
            loss = nn.NLLLoss2d()
            lossVal = loss(outputs, train_label)
            
            lossVal.backward()
            optimizer.step()
            loss_accum += lossVal.data[0]
            
            if (i + 1) % show == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(e+1, epoch, i+1, batch, loss_accum / 20))
                loss_accum = 0.0
            
        torch.save(net.state_dict(), './net.pth')