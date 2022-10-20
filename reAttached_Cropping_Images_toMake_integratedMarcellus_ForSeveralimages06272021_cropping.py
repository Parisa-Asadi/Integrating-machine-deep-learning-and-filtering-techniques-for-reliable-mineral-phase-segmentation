# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 14:19:03 2021

@author: pza0029
"""
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
#######################

os.chdir(r'K:\New folder\New_05222021\Mancos\Final_plot\set1\predict') #main folde0  #C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\predictedImages_UNet\Y_pred_new  #C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\predicted_images\N12
name= []
set='set1'
#GET THE name of all images in a path
for directory_path in glob.glob(r"K:\New folder\New_05222021\Mancos\Final_plot\set1\predict"): #C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\predicted_images\N12
     for count, value in enumerate(glob.glob(os.path.join(directory_path, "*.tiff"))):
         name.append(value)
#######################
Ngroup= 49 #each 49 images make one 2D slice CT images. Change it. 896/128=7, 7*7=49

i=0
for im in range(int(len(name)/Ngroup)):
    img = np.zeros((896, 896,3), dtype=np.uint8) 
    for r in range(0,896,128):
       for c in range(0,896,128):
           
           img1= cv2.imread(f"U_net_{i}.tiff")
           img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)#cv2 always read BGR so you should convert it
           img[r:r+128, c:c+128,:]=img2
           
           i=i+1
    cv2.imwrite(rf"../Reattached/reatached_{set}_{im}.tiff",cv2.cvtColor(img, cv2.COLOR_BGR2RGB))#cv2 always Write BGR so you should convert it. if it does not work try cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imwrite(rf"../Reattached/reatached_{im}.tiff",cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))#cv2 always Write BGR so you should convert it. if it does not work try cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
