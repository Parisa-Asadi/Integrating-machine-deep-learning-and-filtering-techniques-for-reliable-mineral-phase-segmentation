# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:53:21 2021

@author: pza0029
"""
#https://towardsdatascience.com/image-augmentation-examples-in-python-d552c26f2873
# Image Loading Code used for these examples

import cv2
import numpy as np 
#import pandas as pd 
#import matplotlib.pyplot as plt
import glob
import os
import matplotlib.pyplot as plt
from datetime import datetime

#function
def augment_input_images(main_dir, folder_with_image):
    os.chdir(main_dir) #main folde0
    data_path_image = os.path.join(folder_with_image, '*.tif') 
    image = glob.glob(data_path_image)  
    for f1 in range(len(image)): 
        img = cv2.imread(image[f1])
        name= (image[f1].split('\\')[1]).split(".")[0]
        #img = np.flipud(img)
        # if int(name) < 104: #use this when you have some images that they are filped and some that they do not
        #     img = np.flipud(img)
        flipped_img = np.fliplr(img) #it reverse the columns order
        Vflipped_img = np.flipud(img) #it reverse the rows order
        
        #name= (image[f1].split('\\')[2]).split(".")[0]
        cv2.imwrite(f"{folder_with_image}/augmented1/{name}_LR_labeled.tif",flipped_img)
        cv2.imwrite(f"{folder_with_image}/augmented1/{name}_UD_labeled.tif",Vflipped_img)
        cv2.imwrite(f"{folder_with_image}/augmented1/{name}.tif",img)
      
def augment_labeled_images(main_dir, folder_with_image):
    os.chdir(main_dir) #main folde0
    data_path_image = os.path.join(folder_with_image, '*.tif') 
    image = glob.glob(data_path_image)  
    for f1 in range(len(image)): 
        img = cv2.imread(image[f1]) 
        #img = np.flipud(img)
        flipped_img = np.fliplr(img) #it reverse the columns order
        Vflipped_img = np.flipud(img) #it reverse the rows order
        #name= (image[f1].split('\\')[1]).split(".")[0]
        name= (image[f1].split('\\')[2]).split(".")[0]
        cv2.imwrite(f"{folder_with_image}/augmented1/{name}_LR_labeled.tif",flipped_img)
        cv2.imwrite(f"{folder_with_image}/augmented1/{name}_UD_labeled.tif",Vflipped_img)
        cv2.imwrite(f"{folder_with_image}/augmented1/{name}.tif",img)
      
#####Get data for each CT series #########

# main_dir = r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\test'

# #folder_with_image = 'augmented'
# folder_with_image = "labels/augmented"



#main_dir = r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos'
main_dir =r"Desktop" #"C:\Users\Parisa\Box\Ph.D Civil\My_total_works\work\Paluxy_resolution_ML\data"

# folder_with_image = 'Train'
# augment_input_images(main_dir, folder_with_image)

#folder_with_image = "Train\labels"
folder_with_image = "data"
augment_labeled_images(main_dir, folder_with_image)



# folder_with_image = 'Test'
augment_input_images(main_dir, folder_with_image)

# folder_with_image = "Test\labels"
# augment_labeled_images(main_dir, folder_with_image)


