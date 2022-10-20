# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 16:36:26 2021

@author: pza0029
to read and save each color sepertely
"""

import cv2
import numpy as np 
#import pandas as pd 
#import matplotlib.pyplot as plt
import glob
import os
import pandas as pd 
import matplotlib.pyplot as plt
#########################################

os.chdir(r'K:\New folder\New_05222021\Marcellus\Final_plot\set1-2\predict\New folder') #main folde0    

data_path_image = os.path.join( '*.tiff') 
image = glob.glob(data_path_image)
i=0 
for f1 in range(len(image)):
    img2= cv2.imread(image[f1],0)
    #img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)#cv2 always read BGR so you should convert it
     
    black= (img2 == 0)
    Blue= (img2 == 29)
    green= (img2 == 150)
    red= (img2 == 76)
    white= (img2 == 255)
    # segm6= (label_reshape == 5)
    # segm7= (label_reshape == 6)
    # segm8= (label_reshape == 7)
    # make an empty array same size as our image
    black_segments = np.zeros((img2.shape[0],img2.shape[1], 3))
    Blue_segments = np.zeros((img2.shape[0],img2.shape[1], 3))
    red_segments = np.zeros((img2.shape[0],img2.shape[1], 3))
    green_segments = np.zeros((img2.shape[0],img2.shape[1], 3))
    white_segments = np.zeros((img2.shape[0],img2.shape[1], 3))
    black_segments[black]=(0.5,0.5,0.5)
    Blue_segments[Blue]=(0,0,1)
    white_segments[white]=(1,1,1)
    red_segments[red]=(1,0,0)
    green_segments[green]=(0,1,0)
    
    
    #os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\plot_view\cropped\re_attached')
    #cv2.imwrite("black.tiff",cv2.cvtColor(int(black_segments), cv2.COLOR_BGR2RGB))#cv2 always Write BGR so you should convert it. if it does not work try cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imsave(fr"C:\Users\pza0029\Documents\Marcellus\black\black_{f1}.tiff",black_segments)
    plt.imsave(fr"C:\Users\pza0029\Documents\Marcellus\Blue\Blue_{i}.tiff",Blue_segments)
    plt.imsave(fr"C:\Users\pza0029\Documents\Marcellus\green\green_{i}.tiff",green_segments)
    plt.imsave(fr"C:\Users\pza0029\Documents\Marcellus\red\red_{i}.tiff",red_segments)
    plt.imsave(fr"C:\Users\pza0029\Documents\Marcellus\white\white_{i}.tiff",white_segments )
    i=i+1
    # cv2.imshow('RGB Image',img )