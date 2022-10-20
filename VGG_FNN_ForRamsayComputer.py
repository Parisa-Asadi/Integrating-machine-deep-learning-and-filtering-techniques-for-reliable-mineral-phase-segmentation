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
        img2 = np.flipud(img2) # for marcellus use : img = np.flipud(img)
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

# ------------------------------------------------------- #
################# crete labels 0-4 instead of num models  ####################
# ------------------------------------------------------- #
# make an empty array same size as our image

segm1 = (Y == 76)
segm2 = (Y == 158)
segm3 = (Y == 187)
segm4 = (Y ==188)
segm5 = (Y ==209)
# segm6 =( labels == 176)
# segm7 = (labels== 215)
all_segments = np.zeros((Y.shape[0],1))
all_segments[segm1]=(0)
all_segments[segm2]=(1)
all_segments[segm3]=(2)
all_segments[segm4]=(3)
all_segments[segm5]=(4)
# all_segments[segm6]=(5)
# all_segments[segm7]=(6)
Y_train = all_segments




Y_test1 = Y_test.reshape(-1)


segm1 = (Y_test1 == 76)
segm2 = (Y_test1 == 158)
segm3 = (Y_test1 == 187)
segm4 = (Y_test1 ==188)
segm5 = (Y_test1 ==209)
# segm6 =( labels == 176)
# segm7 = (labels== 215)
all_segments = np.zeros((Y_test1.shape[0],1))
all_segments[segm1]=(0)
all_segments[segm2]=(1)
all_segments[segm3]=(2)
all_segments[segm4]=(3)
all_segments[segm5]=(4)
# all_segments[segm6]=(5)
# all_segments[segm7]=(6)
Y_test1 = all_segments


# ------------------------------------------------------- #
################# Model in TensorFlow  ####################
# ------------------------------------------------------- #

horsepower = np.array(X)
normalizer = preprocessing.Normalization(input_shape=[1, 64])
normalizer.adapt(horsepower)

print(normalizer.mean.numpy())
first = np.array(X[:1])
with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


model = keras.Sequential()
model.add(normalizer)
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dense(5))
model.summary()

optim = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9,
                                     beta_2=0.999, epsilon=1e-07,
                                     amsgrad=False, name='Adam') 

call1 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False)

# checkpoint_filepath = r'C:\Users\Parisa\Desktop\checkpoint'
# call2 = model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=False,
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True)


model.compile(optimizer=optim,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# model.fit(X_train, Y_train, epochs=10)

X_train = np.array(X).reshape(X.shape[0], 1, 64)
history = model.fit(X_train, Y_train, epochs=100, validation_split = 0.2,
                    callbacks=[call1])
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
X_test=new_model.predict(X_test)
X_test= X_test.reshape(-1, X_test.shape[3]) 
X_test = np.array(X_test).reshape(X_test.shape[0], 1, 64)
predictions2 = np.argmax(probability_model.predict(X_test),axis=2)
predictions1 = probability_model.predict(X_train)#change it based on train or test
predictions122 = np.argmax(predictions1,axis=2)

'''
# to get accuracy fscore .... for train
##############https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(Y_train,predictions122))#change it based on train or test
print(classification_report(Y_train,predictions122))#change it based on train or test
#print(accuracy_score(Y_train, predictions122))#change it based on train or test

#Using built in keras function
from tensorflow.keras.metrics import MeanIoU
n_classes = 5
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(Y_train,predictions122)
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
sklearn.metrics.f1_score(Y_train,predictions122, average='macro') #‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’
'''
##############

print(f'Time: {time.time() - start}')

# ------------------------------------------------------- #
################# After training  #######################
# ------------------------------------------------------- #

test_loss, test_acc = model.evaluate(X_test,  Y_test1, verbose=2)
print('\nTest accuracy:', test_acc)
#probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])



# ------------------------------------------------------- #
################# Save and Load #######################
#https://stackoverflow.com/questions/64541824/keras-giving-low-accuracy-after-loading-model/65237653#comment114123920_64541824
# ------------------------------------------------------- #
from tensorflow.python.keras import losses
os.chdir(r'K:\New folder\New_05222021\Marcellus')
os.makedirs("saved_VGGFNN_total")
# #!mkdir -p saved_model1
model.save('saved_VGGFNN_total/my_model')

# new_model = tf.keras.models.load_model(r'saved_model2/my_model.h5')
# new_model.summary()
# probability_model1 = tf.keras.Sequential([new_model, tf.keras.layers.Softmax()])
# predictions2 = np.argmax(probability_model1.predict(X_test),axis=2)

# test_loss1, test_acc1 = new_model.evaluate(X_test,  Y_test, verbose=2)
# print('\nTest accuracy:', test_acc1)

########
os.chdir(r'K:\New folder\New_05222021\Marcellus') #main_dir=r'K:\New folder\New_05222021\Marcellus\Marcellus' #main folder #K:\New folder\New_05222021\Marcellus\Marcellus #C:\My_works\Mancos&Marcellus\Mancos
from tensorflow.python.keras import losses
os.makedirs("saved_model3_VGG_FNN")
model.save_weights(r"saved_model3_VGG_FNN/my_modelMarcellus-weights")
#_____________________________#########here you can compile the FNN model and load weights to use the trained model to predict x_test:
model1 = keras.Sequential()
model1.add(normalizer)
model1.add(keras.layers.Dense(32, activation = 'relu'))
model1.add(keras.layers.Dense(5))
model1.summary()

optim = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9,
                                     beta_2=0.999, epsilon=1e-07,
                                     amsgrad=False, name='Adam') 

call1 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False)

# checkpoint_filepath = r'C:\Users\Parisa\Desktop\checkpoint'
# call2 = model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=False,
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True)


model1.compile(optimizer=optim,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#new_model1 = tf.keras.models.load_model(r'saved_model2/my_model.h5')
model1.summary()
#os.chdir("saved_model3")
model1.load_weights("saved_model3_VGG_FNN\my_modelMancos-weights")
test_loss1, test_acc1 = model1.evaluate(X_test,  Y_test, verbose=2)
print('\nTest accuracy:', test_acc1)
#_________________________##############
##################save the trained model using pickle##########################
#https://www.geeksforgeeks.org/saving-a-machine-learning-model/
#https://stackabuse.com/scikit-learn-save-and-restore-models/
# filename = 'totalsaved_model3_VGG_FNN'
# os.makedirs("totalsaved_model3_VGG_FNN")
# import pickle
# pickle.dump(model, open(filename, 'wb'))
# ------------------------------------------------------- #
################# Plot #######################
# ------------------------------------------------------- #
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

############Save history
pd.DataFrame.from_dict(history.history).to_csv("VGG_FNN_features.csv")

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

fig = plt.figure()
ax = fig.add_axes([0.13, 0.13, 1, 1])
ax.plot(hist['epoch'], hist['accuracy'])
ax.set_xlabel('Epoch ')
ax.set_ylabel('Accuracy')

plt.show()


predictions = probability_model.predict(X_test)

# ------------------------------------------------------- #
################# confusion #######################
# ------------------------------------------------------- #

from sklearn.metrics import confusion_matrix
confusion_S1 = confusion_matrix(Y_test1, predictions2, labels=[0,1,2,3,4])
print (confusion_S1)
import seaborn as sns
# cmap = sns.cubehelix_palette(light=5, as_cmap=True)

confusion_s1_df = pd.DataFrame(confusion_S1)
# confusion_s1_df.index= [0,1,2,3,4]
# confusion_s1_df.columns= [0,1,2,3,4]
res = sns.heatmap(confusion_s1_df, annot=True, fmt="d" )
#res.invert_yaxis()
confusion_s1_df.to_csv('confusion_matrix_VGG_FNN_Features_withoutname.csv')


################### evaluate f_score , iou
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(Y_test,predictions2))#change it based on train or test
print(classification_report(Y_test,predictions2))#change it based on train or test
#print(accuracy_score(Y_test,predictions2))#change it based on train or test

#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y1 = labelencoder.fit_transform(Y_test1)
Y2=labelencoder.fit_transform(predictions2)
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


# ------------------------------------------------------- #
#################recreate the segmented images with FNN ######################
# ------------------------------------------------------- #
os.chdir(r'K:\New folder\New_05222021\Marcellus')#os.chdir(r'K:\New folder\New_05222021\Marcellus') #os.chdir(r'K:\New folder\New_05222021\Mancos')
os.mkdir("predicted_images_VGG_FNN")
os.chdir(r'K:\New folder\New_05222021\Marcellus\predicted_images_VGG_FNN') #main folder #os.chdir(r'K:\New folder\New_05222021\Marcellus')
segm1 = (predictions2 == 0)
segm2 = (predictions2 == 1)
segm3 = (predictions2 == 2)
segm4 = (predictions2 ==3)
segm5 = (predictions2 ==4)
# segm6 =( labels == 176)
# segm7 = (labels== 215)
all_segments = np.zeros((predictions2.shape[0],1))
all_segments[segm1]=(76)
all_segments[segm2]=(158)
all_segments[segm3]=(187)
all_segments[segm4]=(188)
all_segments[segm5]=(209)
# all_segments[segm6]=(5)
# all_segments[segm7]=(6)
predictions22= all_segments

Num=21 #number of images 
NumArray = int(len(predictions22)/Num)
for i in range(0,Num):
    predictions3= predictions22[NumArray*i:(i+1)*NumArray]
    final= np.uint8(predictions3.reshape(994,969))#994,968 for mancos . 994,969 for marcellus
    #cv2.imwrite(f"K_mean_1dimention/label_reshape{name}.tif", res2)
    cv2.imwrite(f"VGG_FNN_segmentedimage_{i}.tif",final)

