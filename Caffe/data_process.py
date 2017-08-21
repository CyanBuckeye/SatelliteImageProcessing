import numpy as np
import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import sys
import os
import h5py
from random import shuffle

def load_image(filename, color=True):
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img
    
    
bench = np.zeros((3,))#if mask[i][j] == [1.0,1.0,1.0], it is not a building
bench[0] = 1.0
bench[1] = 1.0
bench[2] = 1.0


image_path = r''#image root
mask_name = ''#mask img
origin_name = ''#rgb img

mask = load_image(os.path.join(image_path, mask_name))#get label
origin = load_image(os.path.join(image_path, origin_name))# get Data

pos_count = 0
neg_count = 0

length = mask.shape[0]
width = mask.shape[1]
stride = 1
kernel_size = 128
std = 20


def Gaussian_std(patch, vec, std = 0.1):
    kernel_size = patch.shape[0]
    channel = patch.shape[2]
    total = 0
    for i in range(0, channel):
        total += vec[i] ** 2

    for i in range(kernel_size):
        for j in range(kernel_size):
             diff = 0
             for k in range(0, channel):
                 diff += (patch[i][j][k] - vec[k]) ** 2
             ratio = max(1 - float(diff) / total, 0)
             patch[i][j] = ratio*patch[i][j]
    return patch

def Gaussian_kernel(kernel_size, std):
    matrix = np.ndarray((kernel_size, kernel_size, 3))
    x0 = y0 = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
             val = std * np.exp(-((i - x0)**2 / 2.0 + (j - y0)**2 / 2.0))
             matrix[i][j][0] = val
             matrix[i][j][1] = val
             matrix[i][j][2] = val
    return matrix

for i in range(kernel_size//2, length - kernel_size//2, stride):
    for j in range(kernel_size//2, width - kernel_size//2, stride):
        if (mask[i][j] == bench).all():
            neg_count = neg_count + 1
        else:
            pos_count = pos_count + 1

train_group_num = 16
test_group_num = 10 
train_ratio = 0.8   
count = min([pos_count, neg_count])       
train_count = int(np.floor(count * train_ratio // train_group_num))
train_num = int(0.9 * train_count)
val_num = int(0.1 * train_count)


test_count = int(np.floor(count * (1 - train_ratio) // test_group_num))


data_root = r''

gaussian_kernel = Gaussian_kernel(kernel_size, std)#try using Gaussian kernel to process each patch


i_begin_pos = kernel_size//2
j_begin_pos = kernel_size//2

i_begin_neg = kernel_size//2
j_begin_neg = kernel_size//2

for temp_index in range(0, train_group_num):
    
    f = h5py.File(data_root + r'\train_data_' + str(temp_index) + '.h5','w')
    f.create_dataset('data', (2*train_num, 3, kernel_size, kernel_size))
    f.create_dataset('label', (2*train_num, 1))
    
    g = h5py.File(data_root + r'\val_data_' + str(temp_index) + '.h5', 'w')
    g.create_dataset('data', (2*val_num, 3, kernel_size, kernel_size))
    g.create_dataset('label', (2*val_num, 1))
    
    general_data = []
    
    
    pos_count = 0
    neg_count = 0

    flag = True
    for i in range(i_begin_pos, length - kernel_size//2, stride): #using sliding-window to go through the whole image
        if flag == False:
            break
        if i == i_begin_pos:
            temp_j_begin = j_begin_pos
        else:
            temp_j_begin = kernel_size // 2
            
        for j in range(temp_j_begin, width - kernel_size//2, stride):
            i_begin_pos = i
            j_begin_pos = j
            
            if pos_count >= train_count:
                flag = False
                break
           
            if (mask[i][j] == bench).all():
                continue
            else: #if the pixel belongs to the expected label, extract the patch surrounding that pixel
                patch = origin[i - kernel_size//2:i + kernel_size//2, 
				j - kernel_size//2:j + kernel_size//2].copy()
                general_data.append((patch * gaussian_kernel, 1))
                pos_count = pos_count + 1
    
    flag = True
    for i in range(i_begin_neg, length - kernel_size//2, stride):
        if flag == False:
            break
        if i == i_begin_neg:
            temp_j_begin = j_begin_neg
        else:
            temp_j_begin = kernel_size // 2
        for j in range(temp_j_begin, width - kernel_size//2, stride):
            i_begin_neg = i
            j_begin_neg = j
            
            if neg_count >= train_count:
                flag = False
                break
            
            if (mask[i][j] == bench).all():
                patch = origin[i - kernel_size//2:i + kernel_size//2, 
				j - kernel_size//2:j + kernel_size//2].copy()
                general_data.append((patch * gaussian_kernel, 0))
                neg_count = neg_count + 1
                
            
    
    shuffle(general_data)
 
    for i in range(0, train_num*2):
        f['data'][i] = general_data[i][0].transpose(2,0,1)
        f['label'][i] = general_data[i][1]
    f.flush()
    f.close()
    
    for i in range(2*train_num, 2*(train_num + val_num)):
        index = i - 2*train_num
        g['data'][index] = general_data[i][0].transpose(2,0,1)
        g['label'][index] = general_data[i][1]
    g.flush()
    g.close()
	

test_pos_i = i_begin_pos
test_pos_j = j_begin_pos

test_neg_i = i_begin_pos
test_neg_j = j_begin_pos
	
for temp_index in range(0, test_group_num):
    h = h5py.File(data_root + r'\test_data_' + str(temp_index) + '.h5','w')
    h.create_dataset('data', (2*test_count, 3, kernel_size, kernel_size))
    h.create_dataset('label', (2*test_count, 1))
    h.create_dataset('coordinate', (2*test_count,2))
    general_data = []
    count = 0
    flag = True
    start_pos_point = test_pos_i
    for i in range(start_pos_point, length - kernel_size//2):
        if flag == False: break
        pos_j_begin = test_pos_j
        if i != start_pos_point:
            pos_j_begin = kernel_size // 2
        for j in range(pos_j_begin, width - kernel_size//2):
            test_pos_i = i
            test_pos_j = j
            if count >= test_count:
                flag = False
                break
            if (mask[i][j] == bench).all():
                continue
            else:
                patch = origin[i - kernel_size//2:i + kernel_size//2, j - kernel_size//2:j + kernel_size//2].copy()
                general_data.append((patch*gaussian_kernel, 1, (i, j)))
                count += 1
    count = 0
    flag = True   
    start_neg_point = test_neg_i
    for i in range(start_neg_point, length - kernel_size//2):
        if flag == False: break
        neg_j_begin = test_neg_j
        if i != start_neg_point:
            neg_j_begin = kernel_size // 2
        for j in range(neg_j_begin, width - kernel_size//2):
            test_neg_i = i
            test_neg_j = j
            if count >= test_count:
                flag = False
                break
            if (mask[i][j] == bench).all():
                patch = origin[i - kernel_size//2:i + kernel_size//2, j - kernel_size//2:j + kernel_size//2].copy()
                general_data.append((patch*gaussian_kernel, 0, (i, j)))
                count += 1
            
    shuffle(general_data)
 
    for i in range(0, len(general_data)):
        h['data'][i] = general_data[i][0].transpose(2,0,1)
        h['label'][i] = general_data[i][1]
        h['coordinate'][i] = np.array([general_data[i][2][0], general_data[i][2][1]])
    
    h.flush()
    h.close()
