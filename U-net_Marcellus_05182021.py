# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:43:25 2021

@author: pza0029
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:50:13 2021

@author: pza0029
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:54:00 2021

@author: pza0029
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:38:00 2021

@author: pza0029
"""
# https://youtu.be/XyX5HNuv-xE
"""
Author: Dr. Sreenivas Bhattiprolu
Multiclass semantic segmentation using U-Net
Including segmenting large images by dividing them into smaller patches 
and stiching them back
To annotate images and generate labels, you can use APEER (for free):
www.apeer.com 
"""

#####set directory to the place that I have saved my simple u_net code

#####set directory to the place that I have saved my simple u_net code
import os
os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation') #main folder
#####
#from simple_multi_unet_model import multi_unet_model #Uses softmax 
from simple_multi_unet_model_Original import multi_unet_model #Uses softmax 
from keras.utils import normalize
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow
#import matplotlib
import kerastuner as kt




#os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1') #main folder
#Resizing images, if needed
SIZE_X = 128 
SIZE_Y = 128
n_classes=5 #Number of classes for segmentation

#Capture training image info as a list
train_images = []

for directory_path in glob.glob(r"C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\Train\augmented1\cropped"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path)       
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
  
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#del path to make sure they will not overwrite
del directory_path
del img_path
del img 
#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob(r"C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\Train\labels\augmented1\cropped"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)       
        #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)
        
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)

#del path to make sure they will not overwrite
del directory_path
del mask_path
del mask
###############################################use this part if your test and train is seperated otherwise comment it
#creating test images:
#Capture testing image info as a list
test_images = []

for directory_path in glob.glob(r"C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\Test\augmented1\cropped"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path)       
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        test_images.append(img)
       
#Convert list to array for machine learning processing        
test_images = np.array(test_images)

#del path to make sure they will not overwrite
del directory_path
del img_path
del img

#Capture mask/label info as a list
test_masks = [] 
for directory_path in glob.glob(r"C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\Test\labels\augmented1\cropped"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)       
        #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        test_masks.append(mask)
        
#Convert list to array for machine learning processing          
test_masks = np.array(test_masks)

#del path to make sure they will not overwrite
del directory_path
del mask_path
del mask
###############################################
###############################################use this part if your test and train is seperated otherwise comment it
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = test_masks.shape
test_masks_reshaped = test_masks.reshape(-1,1)
test_masks_reshaped_encoded = labelencoder.fit_transform(test_masks_reshaped)
test_masks_encoded_original_shape = test_masks_reshaped_encoded.reshape(n, h, w)

np.unique(test_masks_encoded_original_shape)

#################################################



#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

#################################################
#train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

###############################################use this part if your test and train is seperated otherwise comment it
#test_images = np.expand_dims(test_images, axis=3)
test_images = normalize(test_images, axis=1)
test_masks_input = np.expand_dims(test_masks_encoded_original_shape, axis=3)

###############################################


train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training #if your test and train is not seperated. since mine was sepearated I commented this part.
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0, shuffle= False)


###############################################use this part if your test and train is seperated otherwise comment it
#X_train = train_images
X_test = test_images
#y_train = train_masks_input
y_test = test_masks_input
###############################################
##################################################
#del path to make sure they will not overwrite
del train_images
del test_images
del train_masks
del test_masks
del train_masks_input
del test_masks_input
del test_masks_reshaped
del test_masks_encoded_original_shape
del train_masks_reshaped
del train_masks_encoded_original_shape
################################################
###############################################
print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

from tensorflow.keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



yval_masks_cat = to_categorical(y_val, num_classes=n_classes)
yval_cat = yval_masks_cat.reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))
##################################################
#del path to make sure they will not overwrite
#del y_test
#del y_train
###############################################################
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_masks_reshaped_encoded),
                                                 train_masks_reshaped_encoded)
print("Class weights are...:", class_weights)
#############################
#del path to make sure they will not overwrite
del test_masks_reshaped_encoded
del train_masks_reshaped_encoded
#############################
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()




##################################################### Autotuning https://www.tensorflow.org/tutorials/keras/keras_tuner

from simple_multi_unet_model import multi_unet_model as model
metrics1 = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
### https://stackoverflow.com/questions/59439124/keras-tuner-search-function-throws-failed-to-create-a-newwriteablefile-error
tuner = kt.Hyperband(model,
                     objective= kt.Objective("val_f1-score", direction="max"),
                     max_epochs=100,
                     factor=3,
                     directory=os.path.normpath('c:/'),
                     
                     )


stop_early = tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", mode='min', patience=4)

tuner.search(X_train, y_train_cat, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=100, 
                    validation_data=(X_val, yval_cat), 
                    #class_weight=class_weights,
                    shuffle=False,
                    callbacks=[stop_early])
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train_cat, 
                    batch_size = 32, 
                    verbose=1, 
                    epochs=100, 
                    validation_data=(X_val, yval_cat), 
                    #class_weight=class_weights,
                    shuffle=False,
                    callbacks=[stop_early])
                    
model.save('Marcellus_test_fscore_TotalLoss_HyperbandTunning.hdf5')
# _, acc = model.evaluate(X_test, y_test_cat)
# print("Accuracy is = ", (acc * 100.0), "%")

###
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


##################################################### END of Tuning


###################################################### TODO: clean-up the code by copying the ones below to above lines-separate 
#Reused parameters in all models

#n_classes=4
activation='softmax'
# import keras
#LR = 0.0001 old
LR = 0.0001
optim = tensorflow.keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
#dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.2, 0.2, 0.2, 0.2,0.2]))
dice_loss = sm.losses.DiceLoss(class_weights=np.array([16.65390755, 1.52314129, 0.20177158 , 0.32259344, 34.80512586])) #real [16.15390755  1.02314129  1.20177158  0.32259344 34.80512586]
focal_loss = sm.losses.CategoricalFocalLoss(gamma=2)
total_loss = dice_loss + (2 * focal_loss)
#total_loss =  dice_loss
# if int(total_loss)>0:
#     total_loss= total_loss
# else: total_loss=-1*total_loss
# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics1 = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

model.compile(optim, loss= total_loss , metrics=metrics1)

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
########################################################################

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#If starting with pre-trained weights. 
#model.load_weights('???.hdf5')

stop_early = tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", mode='auto', patience=4)                   
history = model.fit(X_train, y_train_cat, 
                    batch_size = 25, 
                    verbose=1, 
                    epochs=300, 
                    validation_data=(X_val, yval_cat), 
                    #class_weight=class_weights,
                    shuffle=False,
                    callbacks=[stop_early])

#model.save('Mancos_test.hdf5')
model.save('Marcellus_final_N12.hdf5')
#model.save('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')
############################################################
#Evaluate the model
	# evaluate model
# _, acc = model.evaluate(X_test, y_test_cat)
# print("Accuracy is = ", (acc * 100.0), "%")

_,_, f11_score = model.evaluate(X_test, y_test_cat)
print("f1-score is = ", (f11_score * 100.0), "%")
###
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


####https://stackoverflow.com/questions/39883331/plotting-learning-curve-in-keras-gives-keyerror-val-acc
#Looks like in Keras + Tensorflow 2.0 val_acc was renamed to val_accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
acc = history.history[metrics1]
val_acc = history.history['val_accuracy']


plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


##################################
# model = get_model()
# model.load_weights('Mancos_test.hdf5') 
#model = get_model()
#model.load_weights('Mancos_test.hdf5')  
#model.load_weights('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')  

#IOU
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)
#np.where(y_pred_argmax==y_test[:,:,:,0],1,0)

import tensorflow 
from tensorflow  import metrics
m = tensorflow.keras.metrics.Accuracy()
m.update_state(y_test[:,:,:,0], y_pred_argmax)
m.result().numpy()
#tensorflow.keras.metrics.Accuracy(y_test[:,:,:,0].reshape(-1,1), y_pred_argmax.reshape(-1,1))
tensorflow.math.confusion_matrix(y_test[:,:,:,0].reshape(-1), y_pred_argmax.reshape(-1),name=True)
##################################################

#Using built in keras function
from tensorflow.keras.metrics import MeanIoU
n_classes = 5
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
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
import sklearn
Y1=y_test.reshape(-1)
Y2=y_pred_argmax.reshape(-1)
sklearn.metrics.f1_score(Y1, Y2, average='macro')
# plt.imshow(train_images[0, :,:,0], cmap='gray')
# plt.imshow(train_masks[0], cmap='gray')
#######################################################################
#Predict on a few images
# model = get_model()
# model.load_weights('Mancos_test.hdf5')  
import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
#test_img_norm=test_img[:,:,:][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()

plt.imshow(predicted_img, cmap='gray')
#plt.imsave('images/test_images/360_segmented.jpg', prediction_image, cmap='gray')
#print(f'Time: {time.time() - start}')
os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\predicted_images\N12') #main folder

for i in range(y_pred_argmax.shape[0]):
    final= y_pred_argmax[i]
    #cv2.imwrite(f"K_mean_1dimention/label_reshape{name}.tif", res2)
    cv2.imwrite(f"U_net_{i}.tif",final)
    
os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\predicted_images\N12') #main folder
 
for i in range(y_pred_argmax.shape[0]):
    final= y_pred_argmax[i]
    #cv2.imwrite(f"K_mean_1dimention/label_reshape{name}.tif", res2)
    segm1 = (final == 0)
    segm2 = (final == 1)
    segm3 = (final == 2)
    segm4 = (final ==3)
    segm5 = (final ==4)
    # segm6 =( labels == 176)
    # segm7 = (labels== 215)
    all_segments = np.float32(np.zeros((final.shape[0],final.shape[1],3)))
    all_segments[segm1]=(0,1,0)
    all_segments[segm2]=(0,0,1)
    all_segments[segm3]=(0,0,0)
    all_segments[segm4]=(1,0,0)
    all_segments[segm5]=(1,1,1)
# all_segments[segm6]=(5)
# all_segments[segm7]=(6)
    #predictions2= all_segments
    plt.imsave(f"U_net_{i}.tiff",all_segments)
#####################################################################

##################################################################### SAVE HISTORY (loss, val)
#df_loss_accuraccy = pd.DataFrame([])
#df_loss_accuraccy.to_csv("df_loss_accuraccy.csv")
pd.DataFrame.from_dict(history.history).to_csv("loss_n12_Marcellus.csv")
#######################################################################
################For test1_images
#test1_images

test1_images = [] #to provide and image to show

for directory_path in glob.glob(r"C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\plot_view\Cropped"): #C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\plot_view\cropped
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path)       
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        test1_images.append(img)
  
#Convert list to array for machine learning processing        
test1_images = np.array(test1_images)
test1_images = normalize(test1_images, axis=1)

##########
y_pred1=model.predict(test1_images)
y_pred1_argmax=np.argmax(y_pred1, axis=3)
os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\plot_view\Cropped\predict') #main folder #C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Mancos\plot_view\cropped\predict
 
for i in range(y_pred1_argmax.shape[0]):
    final= y_pred1_argmax[i]
    #cv2.imwrite(f"K_mean_1dimention/label_reshape{name}.tif", res2)
    segm1 = (final == 0)
    segm2 = (final == 1)
    segm3 = (final == 2)
    segm4 = (final ==3)
    segm5 = (final ==4)
    # segm6 =( labels == 176)
    # segm7 = (labels== 215)
    all_segments = np.float32(np.zeros((final.shape[0],final.shape[1],3)))
    all_segments[segm1]=(0,1,0)
    all_segments[segm2]=(0,0,1)
    all_segments[segm3]=(0,0,0)
    all_segments[segm4]=(1,0,0)
    all_segments[segm5]=(1,1,1)
# all_segments[segm6]=(5)
# all_segments[segm7]=(6)
    #predictions2= all_segments
    plt.imsave(f"U_net_{i}.tiff",all_segments)
####################################################
####################################################


#####################################################################

#Predict on large image

#Apply a trained model on large image

from patchify import patchify, unpatchify

large_image = cv2.imread('large_images/large_image.tif', 0)
#This will split the image into small images of shape [3,3]
patches = patchify(large_image, (128, 128), step=128)  #Step=256 for 256 patches means no overlap

predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        print(i,j)
        
        single_patch = patches[i,j,:,:]       
        single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
        single_patch_input=np.expand_dims(single_patch_norm, 0)
        single_patch_prediction = (model.predict(single_patch_input))
        single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]

        predicted_patches.append(single_patch_predicted_img)

predicted_patches = np.array(predicted_patches)

predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 128,128) )

reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
plt.imshow(reconstructed_image, cmap='gray')
#plt.imsave('data/results/segm.jpg', reconstructed_image, cmap='gray')

plt.hist(reconstructed_image.flatten())  #Threshold everything above 0

# final_prediction = (reconstructed_image > 0.01).astype(np.uint8)
# plt.imshow(final_prediction)

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(large_image, cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(reconstructed_image, cmap='jet')
plt.show()








