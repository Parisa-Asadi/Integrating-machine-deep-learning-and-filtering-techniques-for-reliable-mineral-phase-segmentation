# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:45:47 2021

@author: Parisa

# https://youtu.be/ccdssX4rIh8

Image flips via the horizontal_flip and vertical_flip arguments.
Image rotations via the rotation_range argument

"""

from keras.preprocessing.image import ImageDataGenerator
from skimage import io

# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor. 
#https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        horizontal_flip=True,
        vertical_flip=True)    #Also try nearest, constant, reflect, wrap


######################################################################
##### read image directory and get the list of images ################
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




####----------------------------------------------------####
#### ------------reading the data files-----------------####
####----------------------------------------------------####



datetime_1 = datetime.now()


# ----------------------------------------------------------------- #
##  function for integrating the images and creating the data set ###
# ---------------------------------------------------------------- #
#####function####

 
#always check if it is .tiff or .tif


def Augmentation_total(main_dir, folder_with_image):
    os.chdir(main_dir) #main folde0
    data_path_image = os.path.join(folder_with_image, '*.tiff') 
    image = glob.glob(data_path_image)  
    for f1 in range(len(image)): 
        img = cv2.imread(image[f1]) 
        # Reshape the input image because ...
        #x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
        #First element represents the number of images
        img = img.reshape((1, ) + img.shape)  #Array with shape (1, 256, 256, 3)
        seed=1
        i = 0
        name = folder_with_image.split("\\")[0]
        #Loading a single image for demonstration purposes.
        #Using flow method to augment the image
        for batch in datagen.flow(img, batch_size=16,  
                                  save_to_dir=(f'{folder_with_image}/augmented'), 
                                  save_prefix=(f'{name}_{f1}_aug'), 
                                  save_format='tif',seed=seed):
            i += 1
            if i > 5:
                break  # otherwise the generator would loop indefinitely  
    return 


#####Get data augmentation for each CT series #########

main_dir = r'C:\Users\Parisa\Box\Shared with Parisa\CT\Mancos_ziess_test\core1'

folder_with_image = 'Mancos_90_99'
#folder_with_image = "Mancos_90_99\labeled"
Augmentation_total(main_dir, folder_with_image)



#code is finished



####### check it #############



####################################################################
#Multiple images.
#Manually read each image and create an array to be supplied to datagen via flow method
dataset = []

import numpy as np
from skimage import io
import os
from PIL import Image

image_directory = 'test_folder/'
SIZE = 128
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))

x = np.array(dataset)

#Let us save images to get a feel for the augmented images.
#Create an iterator either by using image dataset in memory (using flow() function)
#or by using image dataset from a directory (using flow_from_directory)
#from directory can beuseful if subdirectories are organized by class
   
# Generating and saving 10 augmented samples  
# using the above defined parameters.  
#Again, flow generates batches of randomly augmented images
"""   
i = 0
for batch in datagen.flow(x, batch_size=16,  
                          save_to_dir='augmented', 
                          save_prefix='aug', 
                          save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely  
"""


#####################################################################
#Multiclass. Read dirctly from the folder structure using flow_from_directory

i = 0
for batch in datagen.flow_from_directory(directory='monalisa_einstein/', 
                                         batch_size=16,  
                                         target_size=(256, 256),
                                         color_mode="rgb",
                                         save_to_dir='augmented', 
                                         save_prefix='aug', 
                                         save_format='png'):
    i += 1
    if i > 31:
        break 

#Creates 32 images for each class. 
        
#Once data is augmented, you can use it to fit a model via: fit.generator
#instead of fit()
#model = 
#fit model on augmented data
#model.fit_generator(datagen.flow(x))