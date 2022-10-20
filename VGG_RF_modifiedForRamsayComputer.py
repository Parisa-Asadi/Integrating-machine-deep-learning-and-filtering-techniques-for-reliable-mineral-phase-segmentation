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
# plt.imshow(X_test[0])
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

####-----------------------------------------------------------------------####
#### -------------------------- RF model implementation--------------------####
####-----------------------------------------------------------------------####

#Redefine X and Y for Random Forest
# X_for_RF = dataset.drop(labels = ['Label'], axis=1)
# Y_for_RF = dataset['Label']

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0, n_estimators=200, min_samples_split=10,n_jobs=5, max_depth=70,class_weight="balanced")

# Train the model on training data
model.fit(X, Y) 

#Save model for future use
os.chdir(r"K:\New folder\New_05222021\Marcellus") #K:\New folder\New_05222021\Marcellus #K:\New folder\New_05222021\Mancos
filename = 'RF_VGG_model_final.sav'
pickle.dump(model, open(filename, 'wb'))

'''
# To calculate accuracy f-score ... for train values 
###########
result1 = model.score(X, Y)
print(result1)
Ypredict1 = model.predict(X)

#from sklearn import metrics

#################https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(Y,Ypredict1))
print(classification_report(Y,Ypredict1))
print(accuracy_score(Y, Ypredict1))
##############
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y1 = labelencoder.fit_transform(Y)
Y2=labelencoder.fit_transform(Ypredict1)
#Using built in keras function
from tensorflow.keras.metrics import MeanIoU
n_classes = 5
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(Y1,Y2)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4]+ values[1,0]+ values[2,0]+ values[3,0]+ values[4,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3]+ values[1,4] + values[0,1]+ values[2,1]+ values[3,1]+ values[4,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4]+ values[0,2]+ values[1,2]+ values[3,2]+ values[4,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2]+ values[3,4] + values[0,3]+ values[1,3]+ values[2,3]+ values[4,3])
class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[4,3]+ values[0,4]+ values[1,4]+ values[2,4]+ values[3,4])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)
print("IoU for class5 is: ", class5_IoU)
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
import sklearn
sklearn.metrics.f1_score(Y1, Y2, average='macro') #‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’
#################
'''


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
del X_test

# prediction = loaded_model.predict(X_test_feature)
prediction = model.predict(X_test_feature)

Y1_test = Y_test.reshape(-1)
result = model.score(X_test_feature , Y1_test)
print(result)

#View and Save segmented image
prediction_image = prediction.reshape(Y_test.shape)
plt.imshow(prediction_image[0], cmap='gray')
#plt.imsave('images/test_images/360_segmented.jpg', prediction_image, cmap='gray')
print(f'Time: {time.time() - start}')
os.mkdir("VGG_RF_results")
os.chdir(r'K:\New folder\New_05222021\Marcellus\VGG_RF_results') #main folder #K:\New folder\New_05222021\Mancos #K:\New folder\New_05222021\Mancos\VGG_RF_results

for i in range(prediction_image.shape[0]):
    final= prediction_image[i]
    #cv2.imwrite(f"K_mean_1dimention/label_reshape{name}.tif", res2)
    cv2.imwrite(f"RF_VGG_features_segmentedimage_{i}_{str(datetime_1).split(' ')[0]}.tif",final)

os.mkdir("VGG_RF_results_recolor")
os.chdir(r'K:\New folder\New_05222021\Marcellus\VGG_RF_results\VGG_RF_results_recolor') #K:\New folder\New_05222021\Mancos\VGG_RF_results\VGG_RF_results_recolor

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
confusion_S1 = confusion_matrix(Y1_test, prediction,labels=[76, 158, 187, 188, 209])
print (confusion_S1)
import seaborn as sns
# cmap = sns.cubehelix_palette(light=5, as_cmap=True)

confusion_s1_df = pd.DataFrame(confusion_S1)
# confusion_s1_df.index= ['158.0', '187.0', '188.0', '209.0', '76.0']
# confusion_s1_df.columns= ['158.0', '187.0', '188.0', '209.0', '76.0']
res = sns.heatmap(confusion_s1_df, annot=True, fmt="d" )
#res.invert_yaxis()
confusion_s1_df.to_csv('confusion_matrix_RF_VGG_Features_test_withoutname.csv')

######evaluate IOU 
###################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y1 = labelencoder.fit_transform(Y1_test)
Y2=labelencoder.fit_transform(prediction)
#Using built in keras function
from tensorflow.keras.metrics import MeanIoU
n_classes = 5
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(Y1,Y2)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4]+ values[1,0]+ values[2,0]+ values[3,0]+ values[4,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3]+ values[1,4] + values[0,1]+ values[2,1]+ values[3,1]+ values[4,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4]+ values[0,2]+ values[1,2]+ values[3,2]+ values[4,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2]+ values[3,4] + values[0,3]+ values[1,3]+ values[2,3]+ values[4,3])
class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[4,3]+ values[0,4]+ values[1,4]+ values[2,4]+ values[3,4])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)
print("IoU for class5 is: ", class5_IoU)
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
import sklearn
sklearn.metrics.f1_score(Y1, Y2, average='macro') #‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’


