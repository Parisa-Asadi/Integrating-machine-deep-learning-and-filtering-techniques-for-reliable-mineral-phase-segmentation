# -*- coding: utf-8 -*-
"""
Created on Thu May 27 09:47:07 2021
#RF1 modified from RF14
@author: pza0029
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:02:14 2020

@author: pza0029
"""



  ###########################################
################ Machine Learning#######################
        ###########################################

## https://www.tensorflow.org/tutorials/keras/classification
# TensorFlow and tf.keras
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
print(tf.__version__) # should be > 2.0
import cv2
import os
import glob 
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
import pickle



# ------------------------------------------------------- #
#################   Saving Big File      ####################
# ------------------------------------------------------- #

# import h5py
# f1 = h5py.File("data_DF.hdf5", "w")
# dset1 = f1.create_dataset("dataset_01", (image.shape[0], 10), dtype='int32', data=image)
# f1.close()
# f2 = h5py.File('data_DF.hdf5', 'r')

import time

start = time.time()

#path = r'C:\Users\pza0029\Box\Shared with Parisa\CT\Marcellus_ziess_test\labled\RF&FFNN'
# path = ""
# path_save_load_test_labels = os.path.join(path, "test\labels\Test_labels.pkl")
# path_save_load_test_features = os.path.join(path, "test\Test_feature.pkl")
# path_save_load_train_labels = os.path.join(path, "train\labels\Train_labels.pkl")
# path_save_load_train_features = os.path.join(path, "train\Train_feature.pkl")
# #load it later or use it in RF or FNN
# Y_test = pd.read_pickle(path_save_load_test_labels)
# X_test = pd.read_pickle(path_save_load_test_features)
# Y_train  = pd.read_pickle(path_save_load_train_labels)
# X_train = pd.read_pickle(path_save_load_train_features)

path = (r"C:\My_works\Mancos&Marcellus\Marcellus")#K:\New folder\New_05222021\Marcellus
os.chdir(path)
Y_test = pd.read_pickle("Test_labels.pkl")
X_test = pd.read_pickle("Test_feature.pkl")
Y_train  = pd.read_pickle("Train_labels.pkl")
X_train = pd.read_pickle("Train_feature.pkl")

X_test = X_test['Original Image'] 
X_train= X_train['Original Image']
# ------------------------------------------------------- #
#################   Preprocessing      ####################
# ------------------------------------------------------- #
# f = 0.3
# image = image[:int(f*image.shape[0])]
# df = pd.read_csv('0.34/converted_label.csv')
# del df['Unnamed: 0']
# df = df.astype('int32')

# y = np.array(df).reshape(-1, 1)
# image = cv2.imread("0.34/Pa-5048_0.34.tif", -1)
# x = image.reshape(-1,1)
# data = np.append(x, y, axis=1)
# train, test = train_test_split(data, test_size=0.3,No doc
#                                random_state=0, shuffle=True)


# X_train = X_train/ 255.0
# Y_train = Y_train.astype('uint8')
# X_test = X_test / 255.0
# Y_test = Y_test.astype('uint8')

# X_train = X_train[:5000]
# Y_train = Y_train[:1000]

# X_train = X_train.reshape(X_train.shape[0], 1, 1)
# X_test = X_test.reshape(X_test.shape[0], 1, 1)

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
all_segments = np.zeros((Y_train.shape[0],Y_train.shape[1]))
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
all_segments = np.zeros((Y_test.shape[0],Y_test.shape[1]))
all_segments[segm1]=(0)
all_segments[segm2]=(1)
all_segments[segm3]=(2)
all_segments[segm4]=(3)
all_segments[segm5]=(4)
# all_segments[segm6]=(5)
# all_segments[segm7]=(6)
Y_test = all_segments


# ------------------------------------------------------- #
################# Model in TensorFlow  ####################
# ------------------------------------------------------- #

horsepower = np.array(X_train)
normalizer = preprocessing.Normalization(input_shape=[1, 1])
normalizer.adapt(horsepower)

print(normalizer.mean.numpy())
first = np.array(X_train[:1])
with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


model = keras.Sequential()
model.add(normalizer)
model.add(keras.layers.Dense(32, activation = 'relu'))
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

X_train = np.array(X_train).reshape(X_train.shape[0], 1, 1)
history = model.fit(X_train, Y_train, epochs=100, validation_split = 0.2,
                    callbacks=[call1])
from tensorflow.python.keras import losses
os.makedirs("FNN1_saved_model1")
#!mkdir -p saved_model1
model.save('FNN1_saved_model1/FNN1_my_model')

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

X_test = np.array(X_test).reshape(X_test.shape[0], 1, 1)
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

test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
print('\nTest accuracy:', test_acc)
#probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])



# ------------------------------------------------------- #
################# Save and Load #######################
# ------------------------------------------------------- #
# from tensorflow.python.keras import losses
# os.makedirs("FFNN1_saved_model1")
# #!mkdir -p saved_model1
# model.save('FFNN1_saved_model1/FNN1_my_model')

# new_model = tf.keras.models.load_model(r'FFNN1_saved_model1/FNN1_my_model')
# new_model.summary()
# probability_model1 = tf.keras.Sequential([new_model, tf.keras.layers.Softmax()])
# predictions4 = np.argmax(probability_model1.predict(X_test),axis=2)

# test_loss1, test_acc1 = new_model.evaluate(X_test,  Y_test, verbose=2)
# print('\nTest accuracy:', test_acc1)

########
from tensorflow.python.keras import losses
os.makedirs("saved_model3_FNN1")
model.save_weights(r"saved_model3_FNN1/my_modelMancos_weights")

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
model1.load_weights("saved_model3_FNN1\my_modelMancos_weights")
test_loss1, test_acc1 = model1.evaluate(X_test,  Y_test, verbose=2)
print('\nTest accuracy:', test_acc1)
##################save the trained model using pickle##########################
#https://www.geeksforgeeks.org/saving-a-machine-learning-model/
#https://stackabuse.com/scikit-learn-save-and-restore-models/
# filename = 'FNNmodel_14features.sav'
# import pickle
# pickle.dump(model, open(filename, 'wb'))
# ------------------------------------------------------- #
################# Plot #######################
# ------------------------------------------------------- #
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

############Save history
pd.DataFrame.from_dict(history.history).to_csv("FNN_1features.csv")
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

fig = plt.figure(figsize=(10,10))
ax = fig.add_axes([0.13, 0.13, 1, 1])
ax.plot(hist['epoch'], hist['accuracy'])
ax.set_xlabel('Epoch ')
ax.set_ylabel('Accuracy')
fig.savefig('books_read.png')
plt.show()


predictions = probability_model.predict(X_test)

# ------------------------------------------------------- #
################# confusion #######################
# ------------------------------------------------------- #

from sklearn.metrics import confusion_matrix
confusion_S1 = confusion_matrix(Y_test, predictions2, labels=[0,1,2,3,4])
print (confusion_S1)
import seaborn as sns
# cmap = sns.cubehelix_palette(light=5, as_cmap=True)

confusion_s1_df = pd.DataFrame(confusion_S1)
# confusion_s1_df.index= [0,1,2,3,4]
# confusion_s1_df.columns= [0,1,2,3,4]
res = sns.heatmap(confusion_s1_df, annot=True, fmt="d" )
#res.invert_yaxis()
confusion_s1_df.to_csv('confusion_matrix_FNN1Features_withoutname.csv')


###################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y1 = labelencoder.fit_transform(Y_test)
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
os.mkdir("predicted_images_FNN1")
os.chdir(r'K:\New folder\New_05222021\Marcellus\predicted_images_FNN1') #main folder
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
# all_segments[segm7]=(6)○
predictions22= all_segments

Num=21 #number of images 
NumArray = int(len(predictions22)/Num)
for i in range(0,Num):
    predictions3= predictions22[NumArray*i:(i+1)*NumArray]
    final= np.uint8(predictions3.reshape(994,969))#994*968 for Mancos
    #cv2.imwrite(f"K_mean_1dimention/label_reshape{name}.tif", res2)
    cv2.imwrite(f"FNN1_segmentedimage_{i}.tif",final)
