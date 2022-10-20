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


#function
def Crop_images(main_dir, folder_with_image):
    os.chdir(main_dir) #main folde0
    data_path_image = os.path.join(folder_with_image, '*.tif') 
    image = glob.glob(data_path_image)  
    for f1 in range(len(image)): 
        img = cv2.imread(image[f1]) 
        # Reshape the input image because ...
        #x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
        #First element represents the number of images
        # for r in range(0,img.shape[0],128):
        #    for c in range(0,img.shape[1],128):
        name= (image[f1].split('\\')[1]).split(".")[0]
        for r in range(0,896,128):
           for c in range(0,896,128):
               cv2.imwrite(f"{folder_with_image}/cropped/{name}_{r}_{c}_labeled.tif",img[r:r+128, c:c+128,:])
      

#####Get data for each CT series #########

main_dir = r'C:\Users\Parisa\Box\Shared with Parisa\CT\Mancos_ziess_test\core1\Mancos_90_99\labeled'

folder_with_image = 'augmented'
#folder_with_image = "Mancos_90_99\labeled"
Crop_images(main_dir, folder_with_image)

