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
from keras.models import Sequential, Model
from keras.layers import Conv2D
import os
from keras.applications.vgg16 import VGG16
from datetime import datetime
import pickle

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
main_dir=r'K:\New folder\New_05222021\Mancos\data\Mancos' #main folder #C:\My_works\Mancos&Marcellus\Marcellus #C:\My_works\Mancos&Marcellus\Mancos
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
X_train1=X_train[0:10,:,:,:]
#Now, let us apply feature extractor to our training data
X=new_model.predict(X_train1)

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
Y = Y_train.reshape(-1)

#Combine X and Y into a dataframe to make it easy to drop all rows with Y values 0
#In our labels Y values 0 = unlabeled pixels. 
dataset = pd.DataFrame(X)
dataset['Label'] = Y
print(dataset['Label'].unique())
print(dataset['Label'].value_counts())

##If we do not want to include pixels with value 0 
##e.g. Sometimes unlabeled pixels may be given a value 0.
#dataset = dataset[dataset['Label'] != 0]

####-----------------------------------------------------------------------####
#### -------------------------- RF model implementation--------------------####
####-----------------------------------------------------------------------####

#Redefine X and Y for Random Forest
X_for_RF = dataset.drop(labels = ['Label'], axis=1)
Y_for_RF = dataset['Label']

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 40, random_state = 42, min_samples_split=3)

# Train the model on training data
model.fit(X_for_RF, Y_for_RF) 

#Save model for future use
filename = 'RF_VGG_model.sav'
pickle.dump(model, open(filename, 'wb'))

#Load model.... 
#loaded_model = pickle.load(open(filename, 'rb'))

####-----------------------------------------------------------------------####
#### -------------------------- RF model test------------------------------####
####-----------------------------------------------------------------------####
#Test on a different image
#READ EXTERNAL IMAGE...
# test_img = cv2.imread('images/test_images/Sandstone_Versa0360.tif', cv2.IMREAD_COLOR)       
# test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
# test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
# test_img = np.expand_dims(test_img, axis=0)

#predict_image = np.expand_dims(X_train[8,:,:,:], axis=0)
X_test_feature = new_model.predict(X_test)
X_test_feature = X_test_feature.reshape(-1, X_test_feature.shape[3])


# prediction = loaded_model.predict(X_test_feature)
prediction = model.predict(X_test_feature)
Y1 = Y_test.reshape(-1)
result = model.score(X_test_feature , Y1 )
print(result)

#View and Save segmented image
prediction_image = prediction.reshape(Y_test.shape)
plt.imshow(prediction_image[0], cmap='gray')
#plt.imsave('images/test_images/360_segmented.jpg', prediction_image, cmap='gray')
print(f'Time: {time.time() - start}')
os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\VGG_RF_results') #main folder

for i in range(prediction_image.shape[0]):
    final= prediction_image[i]
    #cv2.imwrite(f"K_mean_1dimention/label_reshape{name}.tif", res2)
    cv2.imwrite(f"RF_VGG_features_segmentedimage_{i}_{str(datetime_1).split(' ')[0]}.tif",final)

for i in range(prediction_image.shape[0]):
    final= prediction_image[i]
    #cv2.imwrite(f"K_mean_1dimention/label_reshape{name}.tif", res2)
    segm1 = (final == 76)
    segm2 = (final == 158)
    segm3 = (final == 187)
    segm4 = (final ==188)
    segm5 = (final ==209)
    # segm6 =( labels == 176)
    # segm7 = (labels== 215)
    all_segments = np.float32(np.zeros((final.shape[0],final.shape[1],3)))
    all_segments[segm1]=(0,0,0)
    all_segments[segm2]=(1,0,0)
    all_segments[segm3]=(1,1,1)
    all_segments[segm4]=(0,1,0)
    all_segments[segm5]=(1,0,1)
# all_segments[segm6]=(5)
# all_segments[segm7]=(6)
    #predictions2= all_segments
    plt.imsave(f"VGG_RF_color_segmentedimage_{i}.tiff",all_segments)

###importnce
feature_importances = pd.DataFrame(model.feature_importances_)
feature_importances.to_csv('feature_importances_VGGfeatures.csv')

#############confusion####################
from sklearn.metrics import confusion_matrix
confusion_S1 = confusion_matrix(Y1, prediction,labels=[158, 187, 188, 209, 76])
print (confusion_S1)
import seaborn as sns
# cmap = sns.cubehelix_palette(light=5, as_cmap=True)

confusion_s1_df = pd.DataFrame(confusion_S1)
# confusion_s1_df.index= ['158.0', '187.0', '188.0', '209.0', '76.0']
# confusion_s1_df.columns= ['158.0', '187.0', '188.0', '209.0', '76.0']
res = sns.heatmap(confusion_s1_df, annot=True, fmt="d" )
#res.invert_yaxis()
confusion_s1_df.to_csv('confusion_matrix_RF_VGG_Features_test_withoutname.csv')



