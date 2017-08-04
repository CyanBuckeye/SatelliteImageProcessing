#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:40:24 2017

@author: xu.2727
prepare data for deconvolution network
"""

import numpy as np
import os
from random import shuffle
from PIL import Image
import scipy.io as sio
import pdb    
import pickle


class_map = {
(255,255,255):0,
(0,0,255):1,
(0,255,255):2,
(0,255,0):3,
(255,255,0):4,
(255,0,0):5          
}#mapping from color to class

name_map = {
0:'surface',
1:'Building',
2:'low vegetation',
3:'Tree',
4:'Car',
5:'Background'
}


positive_class = [4]
kernel_size = 128
ratio = 0.9

mask_path = ''#path to label
image_path = ''#path to image
dsm_root = ''#path to dsm feature
db_root = ''#directory you want to store data

img_list = [0, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
start_idx = 1
end_idx = len(img_list)

for img_idx in range(start_idx, end_idx):
    
    print("current Image is %d" %(img_idx))
    origin_name = 'top_mosaic_09cm_area' + str(img_list[img_idx]) + '.tif'
    mask_name = 'top_mosaic_09cm_area' + str(img_list[img_idx]) + '.tif'
    dsm_name = str(img_list[img_idx]) + '_data.mat'
    mask = Image.open(os.path.join(mask_path, mask_name))#get label
    origin = Image.open(os.path.join(image_path, origin_name))# get Data
    origin = np.asarray(origin)
    mask = np.asarray(mask)
    
    dsm_file = sio.loadmat(os.path.join(dsm_root, dsm_name))
    dsm = dsm_file['result']#get dsm
    dsm = dsm * 255
    dsm = dsm.astype(np.uint8)
  
    img_height = mask.shape[0]
    img_width = mask.shape[1]
    
    patch_list = []
    build_list = []
    suf_list = []
    tree_list = []
    veg_list = []
    build_count = 0
    suf_count = 0
    tree_count = 0
    veg_count = 0
    stride = 200
    
    label_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for i in range(img_height):
        for j in range(img_width):
            temp_label = class_map[tuple(mask[i][j])]
            if temp_label == 4 or temp_label == 5:
                temp_label = 4
            label_map[i][j] = temp_label
            
    for i in range(kernel_size, img_height):
        for j in range(kernel_size, img_width):
            gt_label = class_map[tuple(mask[i - kernel_size / 2][j - kernel_size / 2])]
            if gt_label == 0:   
                if suf_count % stride == 0:
                    suf_list.append((i,j))
                suf_count += 1
            if gt_label == 1:
                if build_count % stride == 0:
                    build_list.append((i,j))
                build_count += 1
            if gt_label == 2:
                if veg_count % stride == 0:
                    veg_list.append((i,j))
                veg_count += 1
            if gt_label == 3:
                if tree_count % stride == 0:
                    tree_list.append((i,j))
                tree_count += 1
                
                
    shuffle(build_list)
    shuffle(suf_list)
    shuffle(tree_list)
    shuffle(veg_list)
    count = min(len(build_list), len(suf_list), len(tree_list), len(veg_list))
    training_sample = int(4 * count * ratio)
    val_sample = 4 * count - training_sample
    
    for i in range(count):
        patch_list.append((build_list[i], 1))
    for i in range(count):
        patch_list.append((suf_list[i], 0))
    for i in range(count):
        patch_list.append((veg_list[i], 2))
    for i in range(count):
        patch_list.append((tree_list[i], 3))
        
    shuffle(patch_list)
    
    
    
    trainData = np.zeros((training_sample, 7, kernel_size, kernel_size), dtype = np.uint8)
    valData = np.zeros((val_sample, 7, kernel_size, kernel_size), dtype = np.uint8)
    for idx, val in enumerate(patch_list):
        print(idx)
    
        x = val[0][0]
        y = val[0][1]
        
        
        X_copy = origin[x - kernel_size:x, y - kernel_size:y].copy()
        t_copy = origin[x - kernel_size:x, y - kernel_size:y].copy()
        label_copy = mask[x-kernel_size:x, y - kernel_size:y].copy()
        dsm_copy = dsm[x - kernel_size:x, y - kernel_size:y].copy()
        X_copy = X_copy.transpose(2,0,1)
        if idx < training_sample:
            trainData[idx][:3] = X_copy
            trainData[idx][3:6] = dsm_copy.transpose(2,0,1)
            trainData[idx][6] = label_map[x - kernel_size:x, y - kernel_size:y].copy()
        
     
        else:
            valData[idx - training_sample][:3] = X_copy
            valData[idx - training_sample][3:6] = dsm_copy.transpose(2,0,1)
            valData[idx - training_sample][6] = label_map[x - kernel_size:x, y - kernel_size:y].copy()
        
    train_dataPath = db_root + r'/' + 'trainData' + str(img_list[img_idx]) + '.p'
    f = open(train_dataPath, 'wb')
    pickle.dump(trainData, f)
    f.close()
    
    
    val_dataPath = db_root + r'/' + 'valData' + str(img_list[img_idx]) + '.p'
    f = open(val_dataPath, 'wb')
    pickle.dump(valData, f)
    f.close()
    