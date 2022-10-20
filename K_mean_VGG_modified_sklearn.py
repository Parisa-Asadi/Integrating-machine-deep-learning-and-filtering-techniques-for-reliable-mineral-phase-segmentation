# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:30:06 2021
VGG_Kmeans
@author: pza0029
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:59:46 2021

@author: pza0029
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:14:56 2020
parisa asadi
VGG got from:
    https://github.com/bnsreenu/python_for_microscopists/blob/master/159b_VGG16_imagenet_weights_RF_for_semantic.py

helper websites:
    https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
    
Helper Codes:
    1- install Keras:
        conda install -c conda-forge keras
@author: pza0029
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import pickle
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from keras.layers import Conv2D
import os
from keras.applications.vgg16 import VGG16
from datetime import datetime
import tensorflow as tf
# Helper libraries
print(tf.__version__) # should be > 2.0
from tensorflow.keras.layers.experimental import preprocessing



####-----------------------------------------------------------------------####
#### ------------------------reading the data files------------------------####
####-----------------------------------------------------------------------####


#os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1') #main folder
datetime_1 = datetime.now()
import time
start = time.time()

# ------------------------------------------------------------------------------------ #
########  function for integrating the images and creating the data set #############
# ------------------------------------------------------------------------------------ #
#####function####
def integrated_setimage(main_dir, folder_with_image, folder_with_labeledimage ):
    os.chdir(main_dir) #main folde0
   
    #folder_dir_1 = os.listdir(folder_with_image)
    #for i in range(len(folder_dir_1)):
    #img_dir = "" # Enter Directory of all images  
    data_path_image = os.path.join(folder_with_image, '*.tiff')
    data_path_labeled = os.path.join(folder_with_labeledimage, '*.tif')  
    label = glob.glob(data_path_labeled)
    image = glob.glob(data_path_image) 
    #data = [] 
    Y=[]
    X_train= []
    for f1 in range(len(label)): # for marcellus use :    for f1 in range(len(label)-20); #for f1 in range(len(label)):
        img = cv2.imread(label[f1],0) 
        img2 = cv2.imread(image[f1],cv2.IMREAD_COLOR)# for marcellus use : img = np.flipud(img)
        #img2 = np.flipud(img2) # for marcellus use : img = np.flipud(img)
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #img2 = cv2.resize(img2, (SIZE_Y, SIZE_X))
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        Y.append(img)
        X_train.append(img2)
    Y = np.array(Y)
    X_train = np.array(X_train)
    return Y, X_train
######################################################
#change directory
main_dir=r'K:\New folder\New_05222021\Marcellus\Marcellus' #main folder #K:\New folder\New_05222021\Marcellus\Marcellus #C:\My_works\Mancos&Marcellus\Mancos
folder_with_image = 'train'
folder_with_labeledimage = 'train\labels'
Y_train, X_train =  integrated_setimage(main_dir, folder_with_image, folder_with_labeledimage )
#___for test_____
folder_with_image1 = 'test'
folder_with_labeledimage1 = 'test\labels'
Y_test, X_test =  integrated_setimage(main_dir, folder_with_image1, folder_with_labeledimage1 )
# ##############################################################
# ##########read from plk######################################

# path = r"K:\New folder\New_05222021\Mancos\data"
# os.chdir(path)
# Y_test = pd.read_pickle("Test_labels.pkl")
# X_test = pd.read_pickle("Test_feature.pkl")
# Y_train  = pd.read_pickle("Train_labels.pkl")
# X_train = pd.read_pickle("Train_feature.pkl")

# X_test = np.array(X_test['Original Image']) 
# X_train= np.array(X_train['Original Image'])

# ###########################################################
# ###########################################################


#Resizing images is optional, CNNs are ok with large images
SIZE_X = X_test.shape[1] #Resize images (height  = X, width = Y)
SIZE_Y = X_test.shape[2]

# cv2.imshow('img',Y_test[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.figure(figsize=(10,10))
# plt.imshow(Y_test[0])
# plt.show()
####-----------------------------------------------------------------------####
#### ------------------------printing keras VGG16 model--------------------####
####-----------------------------------------------------------------------####
# from keras.applications.vgg16 import VGG16
# from keras.utils.vis_utils import plot_model
# model = VGG16()
# plot_model(model, to_file='vgg.png')


####-----------------------------------------------------------------------####
#### -------------------------------- keras VGG16 model--------------------####
####-----------------------------------------------------------------------####

#Load VGG16 model wothout classifier/fully connected layers
#Load imagenet weights that we are going to use as feature generators
#if you got an error AttributeError: module 'tensorflow.python.framework.ops' has no attribute '_TensorLike', try to bring them from tensorflow instead of keras.
#https://stackoverflow.com/questions/53135439/issue-with-add-method-in-tensorflow-attributeerror-module-tensorflow-python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE_X, SIZE_Y, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0


#After the first 2 convolutional layers the image dimension changes. 
#So for easy comparison to Y (labels) let us only take first 2 conv layers
#and create a new model to extract features
#New model with only first 2 conv layers
new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
new_model.summary()
X_train=X_train[0:20,:,:,:]
# X_train2=X_train[20:40,:,:,:]
# X_train3=X_train[40:59,:,:,:]

#Now, let us apply feature extractor to our training data
X=new_model.predict(X_train)
# X2=new_model.predict(X_train2)
# X3=new_model.predict(X_train3)

del X_train
################################ printing features
#Plot features to view them
# square = 8
# ix=1
# for _ in range(square):
#     for _ in range(square):
#         ax = plt.subplot(square, square, ix)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.imshow(features[0,:,:,ix-1], cmap='gray')
#         ix +=1
# plt.show()

################################ 
#Reassign 'features' as X to make it easy to follow
#X=features
X = X.reshape(-1, X.shape[3])  #Make it compatible for Random Forest and match Y labels

#Reshape Y to match X
Y_train=Y_train[0:20,:,:]
Y = Y_train.reshape(-1)
del Y_train
#Combine X and Y into a dataframe to make it easy to drop all rows with Y values 0
#In our labels Y values 0 = unlabeled pixels. 
# dataset = pd.DataFrame(X)
# dataset['Label'] = Y
# print(dataset['Label'].unique())
# print(dataset['Label'].value_counts())

##If we do not want to include pixels with value 0 
##e.g. Sometimes unlabeled pixels may be given a value 0.
#dataset = dataset[dataset['Label'] != 0]
#######################################