# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 19:32:24 2017

@author: xu.2727

using my pre-trained net to extract features
"""

import caffe
import numpy as np
import os
import scipy.io as sio
from random import shuffle
from PIL import Image
import pdb    

class_map = {
(255,255,255):0,
(0,0,255):1,
(0,255,255):2,
(0,255,0):3,
(255,255,0):4,
(255,0,0):5          
}

name_map = {
0:'surface',
1:'Building',
2:'low vegetation',
3:'Tree',
4:'Car',
5:'Background'
}



kernel_size = 224

mask_path = r'' #path to mask image
image_path = r'' #path to image
mat_root = r'' #path to height information

file_path = r""


output_root = r"" #path to store the extracted features

batch = 10 #batch size
featureLen = 512 # feature dim

img_list = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
start_idx = 0

category_list = ["building", "surface", "vegetationTree"]
for c_idx in range(0, len(category_list)):
    if(os.path.isdir(os.path.join(output_root, category_list[c_idx])) == False):
        os.mkdir(os.path.join(output_root, category_list[c_idx]))
    print(category_list[c_idx])
    if c_idx == 0:
        positive_class = [1]
    if c_idx == 1:
        positive_class = [0]
    if c_idx == 2:
        positive_class = [2,3]
        
    folder_name = os.path.join('Mynetwork', category_list[c_idx])#!
    caffe_model = ""#path to caffe model
    define_file = ""#path to caffe deploy file
    model_file = os.path.join(file_path, caffe_model)
    prototxt_file = os.path.join(file_path, define_file)
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_file, model_file, caffe.TEST)
    extract_layer = 'fc0' #extract features of FC0
    
    for img_idx in range(start_idx, len(img_list)):
    
        print(img_list[img_idx])
        origin_name = 'top_mosaic_09cm_area' + str(img_list[img_idx]) + '.tif'
        mask_name = 'top_mosaic_09cm_area' + str(img_list[img_idx]) + '.tif'
        dsm_path = mat_root + '\\' + str(img_idx) + '_data.mat'
        dsmfile = sio.loadmat(dsm_path)
        dsm = dsmfile['result']
        dsm = dsm * 255
        mask = Image.open(os.path.join(mask_path, mask_name))#get label
        origin = Image.open(os.path.join(image_path, origin_name))# get Data
        origin = np.asarray(origin)
        mask = np.asarray(mask)
        
        img_height = mask.shape[0]
        img_width = mask.shape[1]
        
        patch_list = []
        pos_count = 0
        stride = 100
          
        
        for i in range(kernel_size, img_height):
            for j in range(kernel_size, img_width):
                temp_class = class_map[tuple(mask[i - kernel_size / 2][j - kernel_size / 2])]
                if temp_class in positive_class:
                    
                    if pos_count % stride == 0:
                        patch_list.append(((i,j),temp_class))
                    pos_count += 1

        training_sample = len(patch_list)  
        shuffle(patch_list)
        rd = training_sample / batch
        
        training_sample = rd * batch
        mat_coordfile = os.path.join(output_root, category_list[c_idx], str(img_list[img_idx]) + '_coord.mat')#save coordination of each pixel
        mat_classfile = os.path.join(output_root, category_list[c_idx], str(img_list[img_idx]) + '_class.mat')#save each pixel's corresponding class
        coord = np.zeros((training_sample, 2))
        classid = np.zeros((training_sample,1))
        for idx, val in enumerate(patch_list):
            if idx >= training_sample: break
            coord[idx][0] = val[0][0]
            coord[idx][1] = val[0][1]
            classid[idx] = val[1]
            
        sio.savemat(mat_coordfile, {'coord' : coord})
        sio.savemat(mat_classfile, {'classid' : classid})
        
        feature_file = os.path.join(output_root, category_list[c_idx], str(img_list[img_idx]) + '_featureMap.mat')
        feature_map = np.ones((training_sample, featureLen))
        for temp_rd in range(rd):
            X_train = np.zeros((batch, 6, kernel_size, kernel_size), dtype = np.uint8)#input data
            for j in range(batch):
                idx = temp_rd * batch + j
                x = patch_list[idx][0][0]
                y = patch_list[idx][0][1]
            
            
                X_copy = origin[x - kernel_size:x, y - kernel_size:y].copy()
                X_copy = X_copy[:,:,::-1]
                X_train[j][:3] = X_copy.transpose(2,0,1)
                
            net.forward_all(data = X_train)
            temp_data = net.blobs[extract_layer].data
            temp_data.astype(float) 
            feature_map[temp_rd * batch : temp_rd * batch + batch] = temp_data
        sio.savemat(feature_file, {'feature_map' : feature_map})
