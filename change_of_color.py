# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:41:48 2020

@author: Parisa
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
data_path_image = os.path.join(r"C:\Users\Parisa\Box\Shared with Parisa\CT\Mancos_ziess_test\core1\total", '*.tif')
image = glob.glob(data_path_image) 
for f1 in range(len(image)): 
    img = cv2.imread(image[f1],0)   
        #img= cv2.imread("Classified image0090.tif",0)
        # plt.figure(figsize=(10,10))
        # plt.imshow(img)
        # plt.show()
        
    segm1= (img == 76)
    segm2= (img == 209)
    segm3= (img == 158)
    segm4= (img == 188)
    segm5= (img == 187)
    # segm6= (label_reshape == 5)
    # segm7= (label_reshape == 6)
    # segm8= (label_reshape == 7)
    # make an empty array same size as our image
    all_segments = np.zeros((img.shape[0],img.shape[1], 3))
    all_segments[segm1]=(0,1,0)
    all_segments[segm2]=(1,1,1)
    all_segments[segm3]=(0,0,1)
    all_segments[segm4]=(1,0,0)
    all_segments[segm5]=(0,0,0)
    # all_segments[segm6]=(0,1,0)
    # all_segments[segm7]=(1,1,0)
    # all_segments[segm8]=(1,0,128/255)
    # plt.figure(figsize=(10,10))
    # plt.imshow(all_segments)
    # plt.show()
    plt.imsave(fr"C:\Users\Parisa\Box\Shared with Parisa\CT\Mancos_ziess_test\core1\total\New folder/{f1}.tiff",all_segments)


os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\plot_view\cropped\predict') #main folde0    
i=0
img1= cv2.imread(f"U_net_{i}.tiff")
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)#cv2 always read BGR so you should convert it

os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\plot_view\cropped\re_attached')
cv2.imwrite("reatached.tiff",cv2.cvtColor(img, cv2.COLOR_BGR2RGB))#cv2 always Write BGR so you should convert it. if it does not work try cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


cv2.imshow('RGB Image',img )
cv2.waitkey(0)
 
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()

img2= cv2.imread("reatached.tiff")
plt.figure(figsize=(10,10))
plt.imshow(img)
#plt.show()
plt.savefig("reatached.tiff")
from PIL import Image 
import PIL 
im = Image.fromarray(img)
im.save("geeks.tiff")
