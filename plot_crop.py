# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 21:28:39 2021

@author: pza0029
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 18:53:55 2021

@author: pza0029
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:16:22 2021

@author: Parisa Asadi
Croping the images
"""

############## croping the images ################
############ crop ###################
#https://stackoverflow.com/questions/53755910/how-can-i-liit-a-large-image-into-small-pieces-in-python
#https://stackoverflow.com/questions/55869126/how-to-select-only-a-file-type-with-os-listdir


import cv2
import numpy as np 
#import pandas as pd 
#import matplotlib.pyplot as plt
import glob
import os

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import pickle
from keras.models import Sequential, Model
from keras.layers import Conv2D
import os
from keras.applications.vgg16 import VGG16
from datetime import datetime
import tifffile as tiff

#function
def Crop_Label_images(main_dir, folder_with_image, cut_stride = 512, real_remain_image = True):
    ii=-1
    '''
    
    @Parisa Asadi
    @Date: 11/12/2021
    Parameters
    ----------
    main_dir : TYPE
        main directory that contains several folders.
    folder_with_image : TYPE
        each folder inside the directory.
    cut_stride : TYPE, optional
        the output size of image. The default is 512.

    Returns None
    -------
    None.

    '''
    # import tifffile as tiff
    # a = tiff.imread(image[f1])
    os.chdir(main_dir) #main folde0
    data_path_image = os.path.join(folder_with_image, '*.tif') 
    image = sorted(glob.glob(data_path_image) )
    try:
        os.makedirs(os.path.join(folder_with_image, 'plot'))
    except:
            print('the folder is already created')
    try:
        os.makedirs(os.path.join(folder_with_image, 'plot', 'edges'))
    except:
            print('the folder is already created')            
            
    for f1 in range(len(image)): 
        #img = cv2.imread(image[f1],0)
        #img = tiff.imread(image[f1])
                
        try:
            img = tiff.imread(image[f1])
        except:
            img = cv2.imread(image[f1])
        
        # Reshape the input image because ...
        #x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
        #First element represents the number of images
        # for r in range(0,img.shape[0],128):
        #    for c in range(0,img.shape[1],128):
        name= (image[f1].split('\\')[-1]).split(".")[0]
        
        if real_remain_image == False:
            for r in range(0,img.shape[0], cut_stride):
               for c in range(0,img.shape[1], cut_stride):
                   ii=ii+1
                   if c+cut_stride > img.shape[1]:
                       cv2.imwrite(f"{folder_with_image}/plot/edges/{ii}.tif",img[r:r+cut_stride, img.shape[1]-cut_stride:img.shape[1],:])
                   elif r+cut_stride > img.shape[0]:
                       cv2.imwrite(f"{folder_with_image}/plot/edges/{ii}.tif",img[img.shape[0]-cut_stride:img.shape[0], c:c+cut_stride,:])
                   else:
                       cv2.imwrite(f"{folder_with_image}/plot/{ii}.tif",img[r:r+cut_stride, c:c+cut_stride,:])
                   
        else:
            for r in range(0,img.shape[0], cut_stride):
               for c in range(0,img.shape[1], cut_stride):
                   ii=ii+1
                   if c+cut_stride > img.shape[1]:
                       cv2.imwrite(f"{folder_with_image}/plot/edges/{ii}.tif",img[r:r+cut_stride, c:img.shape[1],:])
                   elif r+cut_stride > img.shape[0]:
                       cv2.imwrite(f"{folder_with_image}/plot/edges/{ii}.tif",img[r:img.shape[0], c:c+cut_stride,:])
                   else:
                       cv2.imwrite(f"{folder_with_image}/plot/{ii}.tif",img[r:r+cut_stride, c:c+cut_stride,:])
                   

#####Get data for each CT series #########
if __name__ =="__main__":
    
    main_dir = r'K:\Parisa\Paluxy_secondProject\data\0p34\RF_label'
    
    for i, folder_with_image in enumerate(glob.glob(r'K:\Parisa\Paluxy_secondProject\data\0p34\RF_label\*')):
        Crop_Label_images(main_dir, folder_with_image, cut_stride = 512, real_remain_image=True)
        # 
##############################################images as input to ML

#function
def Crop_images(main_dir, folder_with_image, cut_stride = 512, real_remain_image = True):
    '''
    
    @Parisa Asadi
    @Date: 11/12/2021
    Parameters
    ----------
    main_dir : TYPE
        main directory that contains several folders.
    folder_with_image : TYPE
        each folder inside the directory.
    cut_stride : TYPE, optional
        the output size of image. The default is 512.

    Returns None
    -------
    None.

    '''
    # import tifffile as tiff
    # a = tiff.imread(image[f1])
    os.chdir(main_dir) #main folde0
    data_path_image = os.path.join(folder_with_image, '*.tif') 
    image = glob.glob(data_path_image) 
    try:
        os.makedirs(os.path.join(folder_with_image, 'cropped'))
    except:
            print('the folder is already created')
    try:
        os.makedirs(os.path.join(folder_with_image, 'cropped', 'edges'))
    except:
            print('the folder is already created')            
            
    for f1 in range(len(image)): 
        #img = cv2.imread(image[f1],0)
        #img = tiff.imread(image[f1])
                
        try:
            img = tiff.imread(image[f1])
            img = img/(np.max(img))
        except:
            img = cv2.imread(image[f1],0)
            img = img/(np.max(img))
        
        # Reshape the input image because ...
        #x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
        #First element represents the number of images
        # for r in range(0,img.shape[0],128):
        #    for c in range(0,img.shape[1],128):
        name= (image[f1].split('\\')[-1]).split(".")[0]
        
        if real_remain_image == False:
            for r in range(0,img.shape[0], cut_stride):
               for c in range(0,img.shape[1], cut_stride):
                   
                   if c+cut_stride > img.shape[1]:
                       cv2.imwrite(f"{folder_with_image}/cropped/edges/{name}_{r}_{c}_labeled.tif",img[r:r+cut_stride, img.shape[1]-cut_stride:img.shape[1]])
                   elif r+cut_stride > img.shape[0]:
                       cv2.imwrite(f"{folder_with_image}/cropped/edges/{name}_{r}_{c}_labeled.tif",img[img.shape[0]-cut_stride:img.shape[0], c:c+cut_stride])
                   else:
                       cv2.imwrite(f"{folder_with_image}/cropped/{name}_{r}_{c}_labeled.tif",img[r:r+cut_stride, c:c+cut_stride])
        else:
            for r in range(0,img.shape[0], cut_stride):
               for c in range(0,img.shape[1], cut_stride):
                   if c+cut_stride > img.shape[1]:
                       cv2.imwrite(f"{folder_with_image}/cropped/edges/{name}_{r}_{c}_labeled.tif",img[r:r+cut_stride, c:img.shape[1]])
                   elif r+cut_stride > img.shape[0]:
                       cv2.imwrite(f"{folder_with_image}/cropped/edges/{name}_{r}_{c}_labeled.tif",img[r:img.shape[0], c:c+cut_stride])
                   else:
                       cv2.imwrite(f"{folder_with_image}/cropped/{name}_{r}_{c}_labeled.tif",img[r:r+cut_stride, c:c+cut_stride])

#####Get data for each CT series #########
if __name__ =="__main__":
    
    main_dir = r'K:\Parisa\Paluxy_secondProject\data\0p34\RF'
    
    for i, folder_with_image in enumerate(glob.glob(r'K:\Parisa\Paluxy_secondProject\data\0p34\RF\*')):
        Crop_images(main_dir, folder_with_image, cut_stride = 512, real_remain_image=True)
        # 
