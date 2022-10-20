# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:07:20 2021

@author: pza0029
"""

import os
os.chdir(r'C:\Users\pza0029\Box\Ph.D Civil\My_total_works\Codes_since_3.8.2021') #main folder
#####
#from simple_multi_unet_model import multi_unet_model #Uses softmax 
from simple_multi_unet_model_Original_64 import multi_unet_model #Uses softmax https://github.com/zhixuhao/unet/blob/master/model.py
from keras.utils import normalize
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow
#import matplotlib
# import kerastuner as kt
import tifffile as tiff


from sklearn.model_selection import train_test_split


#os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1') #main folder
#Resizing images, if needed
SIZE_X = 512
SIZE_Y = 512
n_classes=7 #Number of classes for segmentation
n=len(glob.glob(r"K:\Parisa\Paluxy_secondProject\data\0p34\RF\*"))
#Capture training image info as a list


for i, directory_path in enumerate(glob.glob(r"K:\Parisa\Paluxy_secondProject\data\0p34\RF\*")): #K:\Parisa\Paluxy_secondProject\data\0p34\augmented1 #K:\Parisa\Paluxy_secondProject\data\0p34\only_BSE
    # print(directory_path)
    train_images = []
    for j, img_path in enumerate(glob.glob(os.path.join(directory_path, 'cropped', "*.tif"))):
        # print(img_path)
        
        try:
            img = tiff.imread(img_path)
            img= img/np.max(img)
        except:
            img = cv2.imread(img_path,0)
            img= img/np.max(img)
        #img = cv2.imread(img_path,0)       
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
        
#Convert list to array for machine learning processing      
   
    train_images = np.array(train_images)   
    # train_images = np.expand_dims(train_images, axis=3)
    stacked_data = np.empty((j+1, SIZE_X, SIZE_Y, n))
    stacked_data[:, :, :, i] = train_images


########augmentation
def Augmentation_Big_stack_images(stacked_data):
    stacked_data1 = np.empty(((stacked_data.shape[0])*3, stacked_data.shape[1], stacked_data.shape[2], stacked_data.shape[3]))
    stacked_data1[ 0:stacked_data.shape[0], :, :,:] = stacked_data[:,:,:,:]
     
    flipped_img= stacked_data[:,:, ::-1,:]
    #flipped_img = np.fliplr(stacked_data) #it reverse the columns order
    Vflipped_img = stacked_data[:,::-1, :,:]
    #Vflipped_img = np.flipud(stacked_data) #it reverse the rows order
    stacked_data1[ stacked_data.shape[0]:2*stacked_data.shape[0], :, :,:] = flipped_img[:,:,:,:]
    stacked_data1[ 2*stacked_data.shape[0]:3*stacked_data.shape[0], :, :,:] = Vflipped_img[:,:,:,:]       
    return stacked_data1

stacked_data = Augmentation_Big_stack_images(stacked_data)

train_masks = [] 
for directory_path in glob.glob(r"K:\Parisa\Paluxy_secondProject\data\0p34\RF_label\label\cropped\*.tif"): #K:\Parisa\Paluxy_secondProject\data\0p34\augmented1\augment_256\labels\MineralMap\cropped  #K:\Parisa\Paluxy_secondProject\data\0p34\MineralMap\cropped
    mask = cv2.imread(directory_path, 0)       
    #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
    train_masks.append(mask)

    
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)


def Augmentation_Big_stack_images(stacked_data):
    stacked_data1 = np.empty(((stacked_data.shape[0])*3, stacked_data.shape[1], stacked_data.shape[2]))
    stacked_data1[ 0:stacked_data.shape[0], :, :] = stacked_data[:,:,:]
    
    flipped_img= stacked_data[:,:, ::-1]
    Vflipped_img = stacked_data[:,::-1, :]  
    # flipped_img = np.fliplr(stacked_data) #it reverse the columns order
    # Vflipped_img = np.flipud(stacked_data) #it reverse the rows order
    stacked_data1[ stacked_data.shape[0]:2*stacked_data.shape[0], :, :] = flipped_img[:,:,:]
    stacked_data1[ 2*stacked_data.shape[0]:3*stacked_data.shape[0], :, :] = Vflipped_img[:,:,:]       
    return stacked_data1

train_masks = Augmentation_Big_stack_images(train_masks)

########
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
nn, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(nn, h, w)

np.unique(train_masks_encoded_original_shape)

##
from tensorflow.keras.utils import to_categorical
train_masks_cat = to_categorical(train_masks_encoded_original_shape, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))


y_pred=model.predict(stacked_data)
y_pred_argmax=np.argmax(y_pred, axis=3)
Y_test_max=np.argmax(y_test_cat, axis=3)
#np.where(y_pred_argmax==y_test[:,:,:,0],1,0)