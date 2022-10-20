# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:47:06 2021

@author: Parisa
"""
#import libraries
import numpy as np
import os
import glob
import cv2

#to read and save several images in a 3D array 

os.chdir(r'C:\Users\Parisa\Box\Ph.D Civil\My_total_works\work\Paluxy_resolution_ML\croppedat256') #main folder #K:\New folder\New_05222021\Marcellus\Final_plot\set1 #C:\My_works\Mancos&Marcellus\Marcellus\Train
folder_with_image = ''
data_path = os.path.join(folder_with_image, '*.tif') 
files = glob.glob(data_path) 
#len(files)
arr_label=  np.zeros((223, 472, len(files)))
for i, IM in enumerate(files):
    #image = os.path.join(path, IM)
    #Then you can load it back using:
    df = cv2.imread(IM,0)
    #df = df.reshape(-1,1)
    print(f'the {i} file added')
    arr_label[:,:,i]= df
    #arr_label=np.concatenate((arr_label,df), axis=0)
    print(f'the {i} file added1')
# df.head(2)



#to check the unique values
df = cv2.imread(files[8],-1)
#to check the uniques values are the same in gray and RGB
#1
np.unique(df.reshape(-1, df.shape[2]), axis=0, return_counts=True)
#2
from collections import Counter
Counter([tuple(colors) for i in df for colors in i])
#3
set( tuple(v) for m2d in df for v in m2d )
#np.unique(df, return_counts=True)
######show
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.imshow(df)
plt.title('ee')
plt.show()