# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:45:33 2020

@author: pza0029

Fine Tuning of VGG
"""
from tensorflow.keras.applications import vgg16
import time
import tensorflow as tf
start = time.time()
#from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle
import cv2
import os
import glob 
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D
from keras.applications.vgg16 import VGG16
from datetime import datetime
# from tensorflow.keras import datasets

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


os.chdir(r'C:\Users\pza0029\Shale_project\RF&FNN\VGG') #main folder
datetime_1 = datetime.now()


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
    for f1 in range(len(label)): 
        img = cv2.imread(label[f1],0) 
        img2 = cv2.imread(image[f1],cv2.IMREAD_COLOR)
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
main_dir=r'C:\Users\pza0029\Shale_project\RF&FNN\VGG' #main folder
folder_with_image = 'train'
folder_with_labeledimage = 'train\labels'
Y_train, X_train =  integrated_setimage(main_dir, folder_with_image, folder_with_labeledimage )
#___for test_____
folder_with_image1 = 'test'
folder_with_labeledimage1 = 'test\labels'
Y_test, X_test =  integrated_setimage(main_dir, folder_with_image1, folder_with_labeledimage1 )

#Resizing images is optional, CNNs are ok with large images
SIZE_X = X_test.shape[1] #Resize images (height  = X, width = Y)
SIZE_Y = X_test.shape[2]

cv2.imshow('img',Y_test[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.figure(figsize=(10,10))
plt.imshow(Y_test[0])
plt.show()

# ------------------------------------------------------- #
################# crete labels 0-4 instead of num models  ####################
# ------------------------------------------------------- #
# make an empty array same size as our image

segm1 = (Y_train == 76)
segm2 = (Y_train == 158)
segm3 = (Y_train == 187)
segm4 = (Y_train ==188)
segm5 = (Y_train ==209)
# segm6 =( labels == 176)
# segm7 = (labels== 215)
all_segments = np.zeros(( Y_train.shape[0],Y_train.shape[1],Y_train.shape[2]))
all_segments[segm1]=(0)
all_segments[segm2]=(1)
all_segments[segm3]=(2)
all_segments[segm4]=(3)
all_segments[segm5]=(4)
# all_segments[segm6]=(5)
# all_segments[segm7]=(6)
Y_train = all_segments


segm1 = (Y_test == 76)
segm2 = (Y_test == 158)
segm3 = (Y_test == 187)
segm4 = (Y_test ==188)
segm5 = (Y_test ==209)
# segm6 =( labels == 176)
# segm7 = (labels== 215)
all_segments = np.zeros((Y_test.shape[0],Y_test.shape[1],Y_test.shape[2]))
all_segments[segm1]=(0)
all_segments[segm2]=(1)
all_segments[segm3]=(2)
all_segments[segm4]=(3)
all_segments[segm5]=(4)
# all_segments[segm6]=(5)
# all_segments[segm7]=(6)
Y_test = all_segments


####-----------------------------------------------------------------------####
#### ------------------------VGG fine tuning------------------------####
####-----------------------------------------------------------------------####

vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(SIZE_X, SIZE_Y, 3))

for layer in vgg_conv.layers:
	layer.trainable = False
vgg_conv.summary()  #Trainable parameters will be 0

vgg_conv.get_layer('block1_conv2').trainable=True
vgg_conv.summary()  #Trainable parameters will be 36928


# Create the model
model = Sequential()
# Add the vgg convolutional base model
model.add(vgg_conv)
model.summary()
# Add new layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(5, activation='softmax'))
model.summary()
# Configure the model for training
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit(X_train, Y_train, epochs=10, 
                    validation_data=(X_test, Y_test))

####-----------------------------------------------------------------------####
#### ------------------------plotting the Epoch VS LOSS--------------------####
####-----------------------------------------------------------------------####
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)