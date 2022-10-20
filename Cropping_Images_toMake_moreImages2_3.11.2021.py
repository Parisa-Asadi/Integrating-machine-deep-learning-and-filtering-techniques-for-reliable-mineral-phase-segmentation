# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:16:22 2021

@author: Parisa Asadi
Croping the images
"""

############## croping the images ################
############ crop ###################
#https://stackoverflow.com/questions/53755910/how-can-i-split-a-large-image-into-small-pieces-in-python
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


#function
def Crop_images(main_dir, folder_with_image):
    os.chdir(main_dir) #main folde0
    data_path_image = os.path.join(folder_with_image, '*.tiff') 
    image = glob.glob(data_path_image)  
    for f1 in range(len(image)): 
        img = cv2.imread(image[f1]) 
        # Reshape the input image because ...
        #x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
        #First element represents the number of images
        # for r in range(0,img.shape[0],128):
        #    for c in range(0,img.shape[1],128):
        #name= (image[f1].split('\\')[1]).split(".")[0]
        name= (image[f1]).split(".")[0]
        for r in range(0,896,128):
            for c in range(0,896,128):
                cv2.imwrite(f"cropped/{name}_{r}_{c}_labeled.tif",img[r:r+128, c:c+128,:])
      

#####Get data for each CT series #########

# main_dir = r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\test'

# #folder_with_image = 'augmented'
# folder_with_image = "labels/augmented"



main_dir = r'K:\New folder\New_05222021\Mancos\Final_plot\set1'

folder_with_image = ''
#folder_with_image = "labels/augmented"
Crop_images(main_dir, folder_with_image)




#function
def re_attached_images(main_dir, folder_with_image):
    
    data_path_image = os.path.join(main_dir, '*.tif') 
    os.chdir(r'K:\New folder\New_05222021\Mancos\Final_plot\set1') #C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\plot_view
    image = glob.glob(data_path_image)
    os.chdir(main_dir) #main folde0
    img = np.zeros((896, 896)) 
    for f1 in range(len(image)): 
        # Reshape the input image because ...
        #x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
        #First element represents the number of images
        # for r in range(0,img.shape[0],128):
        #    for c in range(0,img.shape[1],128):
        name= (image[f1].split('\\')[1]).split(".")[0]
        for r in range(0,896,128):
            for c in range(0,896,128):
                img1= cv2.imread(f"predict/{name}_{r}_{c}_labeled.tif")
                img[r:r+128, c:c+128,:]=img1
        cv2.imwrite(f"/re_attached/{image[f1]}.tif",img)
#####Get data for each CT series #########

# main_dir = r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\test'

# #folder_with_image = 'augmented'
# folder_with_image = "labels/augmented"



main_dir = r'K:\New folder\New_05222021\Mancos\Final_plot\set1' #C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\plot_view

folder_with_image = 'predict'
#folder_with_image = "labels/augmented"
re_attached_images(main_dir, folder_with_image)


data_path_image = os.path.join(main_dir, '*.tiff') 
#os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\plot_view') 
image = glob.glob(data_path_image)
os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\plot_view\cropped\predict') #main folde0
img = np.zeros((896, 896,3)) 
for f1 in range(len(image)): 
    # Reshape the input image because ...
    #x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
    #First element represents the number of images
    # for r in range(0,img.shape[0],128):
    #    for c in range(0,img.shape[1],128):
    name= (image[f1].split('\\')[1]).split(".")[0]
    for r in range(0,896,128):
        for c in range(0,896,128):
            img1= cv2.imread(f"cropped/predict/{name}_{r}_{c}_labeled.tiff")
            img[r:r+128, c:c+128,:]=img1
    os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\plot_view\cropped\re_attached')
    cv2.imwrite(f"{name}.tiff",img)
    
os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\plot_view\cropped\predict') #main folde0    
i=0
img = np.zeros((896, 896,3)) 
for r in range(0,896,128):
   for c in range(0,896,128):
       
       img1= cv2.imread(f"U_net_{i}.tiff")
       img[r:r+128, c:c+128,:]=img1
       
       i=i+1
plt.imshow(img) 
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()
os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\plot_view\cropped\re_attached')
cv2.imwrite("reatached.tiff",img)


plt.savefig("reatached.tiff")
from PIL import Image 
import PIL 
im = Image.fromarray(img)
im1 = im.save("geeks.tif")