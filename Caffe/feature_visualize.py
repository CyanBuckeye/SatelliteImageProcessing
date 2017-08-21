# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 09:20:08 2017

@author: xu.2727

visualize extracted features
"""

import scipy.io as sio
import numpy as np
import os
import caffe 
import random
from PIL import Image
from sklearn.decomposition import PCA
#from matplotlib.mlab import PCA
import pdb

file_path = r''
processPath = r''
mat = sio.loadmat(processPath)
indcell = mat['poscell']
dsm_path = r''
dsmfile = sio.loadmat(dsm_path)
dsm = dsmfile['result']
dsm = dsm * 255
kernel_size = 224
category_list = ["building", "surface", "vegetationTree"]
img_list = [0, 1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
start_idx = 0
end_idx = 1
picknum = 10
featureLen = 512
result_path = r'' #path to store the visualization results
for c_idx in range(2, 3):
    print(category_list[c_idx])
    if c_idx == 0:
        positive_class = [1]
    if c_idx == 1:
        positive_class = [0]
    if c_idx == 2:
        positive_class = [2,3]
        
    folder_name = os.path.join('Mynetwork', category_list[c_idx])#!
    caffe_model = ""#path to caffe model
    define_file = ""#path to deploy file
    model_file = os.path.join(file_path, caffe_model)
    prototxt_file = os.path.join(file_path, define_file)
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_file, model_file, caffe.TEST)
    extract_layer = 'fc4'#512-feature
    for img_idx in range(start_idx,1):
    
        print(img_list[img_idx])
        origin_name = r''#path to image
        origin = Image.open(origin_name)
        origin = np.asarray(origin)
        n_origin = np.zeros((origin.shape[0], origin.shape[1], 3))
        n_origin[:,:,:3] = origin
        
        img = np.zeros((origin.shape[0] + kernel_size, origin.shape[1] + kernel_size, 3))
        img[kernel_size / 2 : kernel_size / 2 + origin.shape[0], kernel_size / 2 : kernel_size / 2 + origin.shape[1], :] = n_origin
        mask_name = r''
        mask = Image.open(mask_name)
        mask = np.asarray(mask)
        
        img_height = mask.shape[0]
        img_width = mask.shape[1]
        
        
        training_sample = indcell.shape[1]
        f = np.zeros((training_sample, featureLen))
        print(training_sample)
        for idx in range(training_sample):
            print(idx)
            X_train = np.zeros((picknum, 3, kernel_size, kernel_size), dtype = np.uint8)#input data
            rg = indcell[0][idx].shape[0]
            for j in range(picknum):#randomly pick several points in this segmentation
                pos = random.randint(0, rg - 1)
                x = indcell[0][idx][pos][0] - 1 + kernel_size / 2
                y = indcell[0][idx][pos][1] - 1 + kernel_size / 2
            
            
                X_copy = img[x - (kernel_size / 2):x + (kernel_size / 2), y - (kernel_size / 2):y + (kernel_size / 2)].copy()
                X_copy = X_copy[:,:,::-1]
                X_train[j] = X_copy.transpose(2,0,1)
                
            net.forward_all(data = X_train)
            temp_data = net.blobs[extract_layer].data
            temp_data.astype(float) 
            fea = temp_data.sum(0)
            fea = fea / picknum
            f[idx] = fea
        pca_feature = f.transpose((1,0))
        if c_idx == 1:
            sio.savemat('surface.mat', {'f': pca_feature})
        if c_idx == 2:
            sio.savemat('veg_tree.mat', {'f': pca_feature})
       
        
