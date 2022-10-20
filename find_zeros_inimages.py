# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 21:11:43 2021

@author: pza0029
"""

#####set directory to the place that I have saved my simple u_net code
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

x=[]
y=[]
for i, directory_path in enumerate(glob.glob(r"K:\Parisa\Paluxy_secondProject\data\0p34\RF\*")): #K:\Parisa\Paluxy_secondProject\data\0p34\augmented1 #K:\Parisa\Paluxy_secondProject\data\0p34\only_BSE
    # print(directory_path)
    train_images = []
    for j, img_path in enumerate(glob.glob(os.path.join(directory_path, 'cropped', "*.tif"))):
        # print(img_path)
        
        try:
            img = tiff.imread(img_path)
            img= img
            x.append(img_path)
            y.append(np.max(img))
            # x=(np.max(img))
            # y=img_path
        except:
            img = cv2.imread(img_path,0)
            x.append(img_path)
            y.append(np.max(img))
        #img = cv2.imread(img_path,0)       
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
        
#Convert list to array for machine learning processing      
   
    train_images = np.array(train_images)   
    # train_images = np.expand_dims(train_images, axis=3)
    stacked_data = np.empty((j+1, SIZE_X, SIZE_Y, n))
    stacked_data[:, :, :, i] = train_images
    
x= np.array(x) 
y=np.array(y) 
x = pd.DataFrame(x)
x["y"]=y

x.to_csv('x.csv')