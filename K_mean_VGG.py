# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 23:29:18 2020

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
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16


os.chdir(r'K:\New folder\New_05222021\Mancos\data\Mancos') #main folder

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
main_dir=r'K:\New folder\New_05222021\Mancos\data\Mancos' #main folder
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

#Now, let us apply feature extractor to our training data
X_train=X_train[0:20,:,:,:]
# X_train2=X_train[20:40,:,:,:]
# X_train3=X_train[40:59,:,:,:]

#Now, let us apply feature extractor to our training data
X=new_model.predict(X_train)
# X2=new_model.predict(X_train2)
# X3=new_model.predict(X_train3)

del X_train

################################ printing features
# #Plot features to view them
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

X = X.reshape(X.shape[0],-1, X.shape[3])  #Make it compatible for Random Forest and match Y labels

#Reshape Y to match X
Y = Y_train.reshape( Y_train.shape[0], -1)

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
#### -------------------------- K_mean model implementation--------------------####
####-----------------------------------------------------------------------####
import time
#Redefine X and Y for Random Forest
# X_for_Kmean = dataset.drop(labels = ['Label'], axis=1)
# Y_for_Kmean = dataset['Label']
# X_for_Kmean = X
# Y_for_Kmean = Y
#os.mkdir("K_mean_VGG")
for kk in range(3,7):
    for i in range(X.shape[0]):
        #name = f1.split('.')[0].split('_')[-1]
        #kmean clustering use cv2. it is better for images.
        # convert to np.float32
        #img_reshape_float = img_reshape 
        img_reshape_float = np.float32(X[i])
        #img_reshape_float.max()
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1)
        K = kk #number of cluster
        attempts = 40 #
        '''attempts is : Flag to specify the number of times the algorithm is executed using different /
        initial labellings. The algorithm returns the labels that yield the best compactness. /
        This compactness is returned as output.
        '''
        ret,label,center=cv2.kmeans(img_reshape_float,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS) # or cv2.KMEANS_RANDOM_CENTERS
    
        #get the center
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        #res2 = res.reshape((SIZE_X,SIZE_Y))
        label_reshape = label.reshape((SIZE_X,SIZE_Y))
        cv2.imwrite(rf"K:\New folder\New_05222021\Mancos\data\Mancos\K_mean_VGG\label_reshape{i}_{kk}.tif", label_reshape)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#######################relabel###############################################
os.chdir(r'C:\Users\pza0029\Shale_project\RF&FNN\VGG\K_mean_VGG') #main folder
image = glob.glob(r'C:\Users\pza0029\Shale_project\RF&FNN\VGG\K_mean_VGG')

# plt.figure(figsize=(15,15))
# plt.imshow(img)
#plt.imsave("18.tiff",all_segments)

for f1 in range(len(image)): 
    img = cv2.imread(image[f1],0) 
    #make binary image segments
    segm1= (img == 0)
    segm2= (img == 1)
    segm3= (img == 2)
    segm4= (img == 3)
    segm5= (img == 4)
    segm6= (img == 5)
    segm7= (img == 6)
    segm8= (img == 7)
    # make an empty array same size as our image
    all_segments = np.zeros((image.shape[0],image.shape[1],3))
    all_segments[segm1]=(1,0,0)
    all_segments[segm2]=(1,1,0)
    all_segments[segm3]=(1,1,1)
    all_segments[segm4]=(1,0,1)
    all_segments[segm5]=(0,1,0)
    all_segments[segm6]=(0,0,1)
    all_segments[segm7]=(0,1,1)
    all_segments[segm8]=(0.5,0.5,0.5)

    all_segments  = np.uint8(all_segments )
    cv2.imwrite(f"relable/label_reshaperecon_{KK}_{i}.tif", all_segments)


# plt.figure(figsize=(15,15))
# plt.imshow(all_segments)
# plt.imsave("18.tiff",all_segments)


# cv2.imshow('res2',res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# cv2.imwrite("KMean_labels.tif", res2)

# img5 = cv2.imread("KMean_labels.tif",-1)

#image = img_as_ubyte(image) # it just divide your image to 256

# scale the image to 8 bit . I think it is similar the work the imageJ does
# min = img .min()
# max = img .max()
# img1 =((img  - min)/(max-min)) * 255

# #kmean clustering use sklearn
# kmeans = KMeans(n_clusters=6, init = "K-means++", n_init=10,
#        max_iter=300, tol=1e-4, precompute_distances='auto',
#        verbose=0, random_state=None, copy_x=True,
#        n_jobs=None, algorithm='auto') 

# kmeans.fit(img1)
# labels = kmeans.predict(img1)
# centroids = kmeans.cluster_centers_

# cv2.imshow("labels",labels)
# #cv2.imshow("original",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

########evaluate############################################################################################################
#evaluate. read all of them and then get the accuracy and confusion matrix
##https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import cv2
import glob 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_path = os.path.join(r"C:\Users\pza0029\Box\Shared with Parisa\CT\Marcellus_ziess_test\labled\350_360\K_mean_1dimention\relable", '*.tif') 
data_path2 = os.path.join(r"C:\Users\pza0029\Box\Shared with Parisa\CT\Marcellus_ziess_test\labled\350_360\labels", '*.tif') 
K_mean_results = glob.glob(data_path) 
labled = glob.glob(data_path2) 
Y_test = []
predict=[] 
arr = []
arr.append([1,2,3])
for f1 in range(len(labled)):
    
    #name = f1.split('.')[0].split('_')[-1]
    # read image
    #img = cv2.imread("Auburn_Shale_Lt_btCore_20X_0p8um_bin2_recon005.tiff",-1)
    img = cv2.imread(K_mean_results[f1],0) 
    img2 = cv2.imread(labled[f1],0)
    #make a vector to give it to kmean clustering.
    img_reshape = img.reshape(-1) # if it had more channel ((-1,3)) to have 3 chnnel or 3 vector for each channel.
    img_reshape2 = img2.reshape(-1)
    Y_test =np.append(Y_test , img_reshape2)
    predict=np.append(predict , img_reshape)
   
accuracy_score(Y_test, predict)
confusion_S1 = confusion_matrix(Y_test, predict, labels=[188,87,158,209,76,49])
print (confusion_S1)

# cmap = sns.cubehelix_palette(light=5, as_cmap=True)
confusion_s1_df = pd.DataFrame(confusion_S1)
confusion_s1_df.index= [188,87,158,209,76,49]
confusion_s1_df.columns= [188,87,158,209,76,49]
res = sns.heatmap(confusion_s1_df, annot=True, fmt="d",  )
res.invert_yaxis()
confusion_s1_df.to_csv('confusion_matrix_K_mean_results.csv')
