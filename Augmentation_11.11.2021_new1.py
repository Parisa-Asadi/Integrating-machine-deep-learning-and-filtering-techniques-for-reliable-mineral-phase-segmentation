# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:19:10 2021

@author: pza0029
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:43:25 2021

@author: pza0029
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:50:13 2021
@author: pza0029
# https://youtu.be/XyX5HNuv-xE
"""
######################################################################################
######################## read libraries ##############################################\
#####set directory to the place that I have saved my simple u_net code
import os
os.chdir(r'C:\Users\pza0029\Box\Ph.D Civil\My_total_works\Codes_since_3.8.2021') #main folder
#####
#from simple_multi_unet_model import multi_unet_model #Uses softmax 
from simple_multi_unet_model_Original_64 import multi_unet_model #Uses softmax https://github.com/zhixuhao/unet/blob/master/model.py
from keras.utils import normalize
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow
#import matplotlib
# import kerastuner as kt
import tifffile as tiff
import natsort
from sklearn.model_selection import train_test_split

######################################################################################
######################## get images ##################################################

##################### input images##########################
SIZE_X = 512
SIZE_Y = 512
n_classes=7 #Number of classes for segmentation
n=len(glob.glob(r"K:\Parisa\Paluxy_secondProject\data\0p34\RF\*"))

img_lst2 = glob.glob(r"K:\Parisa\Paluxy_secondProject\data\0p34\RF\*") #K:\Parisa\Paluxy_secondProject\data\0p34\augmented1\augment_256\labels\MineralMap\cropped  #K:\Parisa\Paluxy_secondProject\data\0p34\MineralMap\cropped
img_lst2 = natsort.natsorted(img_lst2,reverse=False)
for i, directory_path in enumerate(img_lst2): #K:\Parisa\Paluxy_secondProject\data\0p34\augmented1 #K:\Parisa\Paluxy_secondProject\data\0p34\only_BSE
    # print(directory_path)
    train_images = []
    for j, img_path in enumerate(glob.glob(os.path.join(directory_path, 'cropped', "*.tif"))):
        # print(img_path)
        try:
            img = tiff.imread(img_path)
            # img= img
            # x.append(img_path)
            # y.append(np.max(img))
            # x=(np.max(img))
            # y=img_path
        except:
            img = cv2.imread(img_path,0)
            # x.append(img_path)
            # y.append(np.max(img))
        #img = cv2.imread(img_path,0)       
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
        
    train_images = np.array(train_images)   
    # train_images = np.expand_dims(train_images, axis=3)
    stacked_data = np.empty((j+1, SIZE_X, SIZE_Y, n))
    stacked_data[:, :, :, i] = train_images
    

##################### split input images#############
train_images, test_images = train_test_split(stacked_data, test_size=0.3, shuffle=False)
test_images, val_images = train_test_split(test_images , test_size=0.5, shuffle=False)


nn,xx,yy,zz=np.shape(train_images)
nnT,xxT,yyT,zzT=np.shape(test_images)

# dd=np.empty((xx,yy,zz))
# dd[:,:,:]=train_images[1,:,:,:]
del directory_path
del img_path
del img 

########augmentation
def Augmentation_Big_stack_images(stacked_data):
    stacked_data1 = np.empty(((stacked_data.shape[0])*3, stacked_data.shape[1], stacked_data.shape[2], stacked_data.shape[3]))
    stacked_data1[ 0:stacked_data.shape[0], :, :,:] = stacked_data[:,:,:,:]
     
    flipped_img= stacked_data[:,:, ::-1,:]
    #flipped_img = np.fliplr(stacked_data) #it reverse the columns order
    Vflipped_img = stacked_data[:,::-1, :,:]
    #Vflipped_img = np.flipud(stacked_data) #it reverse the rows order
    stacked_data1[ stacked_data.shape[0]:2*stacked_data.shape[0], :, :,:] = flipped_img[:,:,:,:]
    stacked_data1[ 2*stacked_data.shape[0]:3*stacked_data.shape[0], :, :,:] = Vflipped_img[:,:,:,:]       
    return stacked_data1
train_images = Augmentation_Big_stack_images(train_images)
test_images = Augmentation_Big_stack_images(test_images)
val_images = Augmentation_Big_stack_images(val_images)
del stacked_data
#stacked_data = Augmentation_Big_stack_images(stacked_data)
####################
#Capture mask/label info as a list
train_masks = [] 
img_lst3 = glob.glob(r"K:\Parisa\Paluxy_secondProject\data\0p34\RF_label\label\cropped\*.tif") #K:\Parisa\Paluxy_secondProject\data\0p34\augmented1\augment_256\labels\MineralMap\cropped  #K:\Parisa\Paluxy_secondProject\data\0p34\MineralMap\cropped
img_lst3 = natsort.natsorted(img_lst3,reverse=False)
for i, directory_path in enumerate(img_lst3):
    mask = cv2.imread(directory_path, 0)       
    #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
    train_masks.append(mask)

    
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)
train_masks, test_masks = train_test_split(train_masks , test_size=0.3, shuffle=False)
test_masks, val_masks = train_test_split(test_masks , test_size=0.5, shuffle=False)
#del path to make sure they will not overwrite
del directory_path
#del mask_path
del mask

nn_mask,xx_mask,yy_mask=np.shape(train_masks)
nnT_mask,xxT_mask,yyT_mask=np.shape(test_masks)

########augmentation
def Augmentation_Big_stack_masks(stacked_data):
    stacked_data1 = np.empty(((stacked_data.shape[0])*3, stacked_data.shape[1], stacked_data.shape[2]))
    stacked_data1[ 0:stacked_data.shape[0], :, :] = stacked_data[:,:,:]
    
    flipped_img= stacked_data[:,:, ::-1]
    Vflipped_img = stacked_data[:,::-1, :]  
    # flipped_img = np.fliplr(stacked_data) #it reverse the columns order
    # Vflipped_img = np.flipud(stacked_data) #it reverse the rows order
    stacked_data1[ stacked_data.shape[0]:2*stacked_data.shape[0], :, :] = flipped_img[:,:,:]
    stacked_data1[ 2*stacked_data.shape[0]:3*stacked_data.shape[0], :, :] = Vflipped_img[:,:,:]       
    return stacked_data1

train_masks = Augmentation_Big_stack_masks(train_masks)
test_masks = Augmentation_Big_stack_masks(test_masks)
val_masks = Augmentation_Big_stack_masks(val_masks)


####################

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(train_masks[0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(train_images[0,:,:,8], cmap='jet')
plt.show()

#cv2.imshow('RGB Image2',train_masks[0])
###############################################
###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
nn, h, w = test_masks.shape
test_masks_reshaped = test_masks.reshape(-1)
test_masks_reshaped_encoded = labelencoder.fit_transform(test_masks_reshaped)

test_masks_encoded_original_shape = test_masks_reshaped_encoded.reshape(nn, h, w)

np.unique(test_masks_encoded_original_shape)

########
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
nn, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(nn, h, w)

np.unique(train_masks_encoded_original_shape)

########
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
nn, h, w = val_masks.shape
val_masks_reshaped = val_masks.reshape(-1)
val_masks_reshaped_encoded = labelencoder.fit_transform(val_masks_reshaped)
val_masks_encoded_original_shape = val_masks_reshaped_encoded.reshape(nn, h, w)

np.unique(val_masks_encoded_original_shape)

###############################################
#################################################
#train_images = np.expand_dims(train_images, axis=3)
#train_images = normalize(train_images, axis=1)
# train_images = train_images
# val_images = val_images
# #test_images = np.expand_dims(test_images, axis=3)
# #test_images = normalize(test_images, axis=1)
# test_images = test_images
###############################################use this part if your test and train is seperated otherwise comment it
X_train = train_images
X_test = test_images
y_train = train_masks_encoded_original_shape
y_test =test_masks_encoded_original_shape
X_val= val_images
y_val = val_masks_encoded_original_shape
###############################################
##################################################
#del path to make sure they will not overwrite
del train_images
del test_images
del train_masks
del test_masks
del test_masks_reshaped
del test_masks_encoded_original_shape
del train_masks_reshaped
del train_masks_encoded_original_shape
del val_masks_reshaped
del val_masks_encoded_original_shape
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
# del y_test
# del y_train
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
# IMG_HEIGHT = stacked_data.shape[1]
# IMG_WIDTH  = stacked_data.shape[2]
# IMG_CHANNELS = stacked_data.shape[3]
def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()




##################################################### Autotuning https://www.tensorflow.org/tutorials/keras/keras_tuner

# from simple_multi_unet_model import multi_unet_model as model
# metrics1 = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
# ### https://stackoverflow.com/questions/59439124/keras-tuner-search-function-throws-failed-to-create-a-newwriteablefile-error
# tuner = kt.Hyperband(model,
#                      objective= kt.Objective("val_f1-score", direction="max"),
#                      max_epochs=100,
#                      factor=3,
#                      directory=os.path.normpath('c:/'),
                     
#                      )


# stop_early = tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", mode='min', patience=4)

# tuner.search(X_train, y_train_cat, 
#                     batch_size = 16, 
#                     verbose=1, 
#                     epochs=100, 
#                     validation_data=(X_val, yval_cat), 
#                     #class_weight=class_weights,
#                     shuffle=False,
#                     callbacks=[stop_early])
# # Get the optimal hyperparameters
# best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# print(f"""
# The hyperparameter search is complete. The optimal number of units in the first densely-connected
# layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
# is {best_hps.get('learning_rate')}.
# """)

# model = tuner.hypermodel.build(best_hps)
# history = model.fit(X_train, y_train_cat, 
#                     batch_size = 32, 
#                     verbose=1, 
#                     epochs=100, 
#                     validation_data=(X_val, yval_cat), 
#                     #class_weight=class_weights,
#                     shuffle=False,
#                     callbacks=[stop_early])
                    
# model.save('Marcellus_test_fscore_TotalLoss_HyperbandTunning.hdf5')
# # _, acc = model.evaluate(X_test, y_test_cat)
# # print("Accuracy is = ", (acc * 100.0), "%")

# ###
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


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
dice_loss = sm.losses.DiceLoss(class_weights=np.array([ 0.58703683,  1.95406471,  4.22859746,  0.2476004, 8.44669133, 47.34786986, 2.70259229])) #real [16.15390755  1.02314129  1.20177158  0.32259344 34.80512586]
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

# model.compile(optim, loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
########################################################################

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#If starting with pre-trained weights. 
#model.load_weights('???.hdf5')

stop_early = tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", mode='auto', patience=10)                   
history = model.fit(X_train, y_train_cat, 
                    batch_size = 3, 
                    verbose=1, 
                    epochs=300, 
                    validation_data=(X_val, yval_cat), 
                    #class_weight=class_weights,
                    shuffle=False,
                    callbacks=[stop_early])

#model.save('Mancos_test.hdf5')
model.save(r'K:\Parisa\Paluxy_secondProject\data\0p34\Project2_11282021_UNet.hdf5')
#model.save('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')
model = get_model()
# model.load_weights(r'K:\Parisa\Paluxy_secondProject\data\0p34\Project2_11282021_UNet.hdf5')
############################################################
import os
# os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm

print('sm.version=' + sm.__version__)
sm.set_framework('tf.keras')

#model = sm.Unet('efficientnetb0', classes=1, input_shape=(128, 128, 3), decoder_filters=(512, 256, 128, 64, 32), activation='sigmoid')

# model = sm.Unet('resnet34', encoder_weights='imagenet')
model = sm.Unet('resnet34', encoder_weights='imagenet', classes=7, activation='softmax')


# from distutils.sysconfig import get_python_lib
# print(get_python_lib('segmentation_models'))


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




################################## validation
 
# model = get_model()
# model.load_weights('Mancos_test.hdf5') 
#model = get_model()
#model.load_weights('Mancos_test.hdf5')  
#model.load_weights('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')  

#IOU
y_pred=model.predict(X_test)#X_test
y_pred_argmax=np.argmax(y_pred, axis=3)
#Y_test_max=np.argmax(y_test_cat, axis=3)

def bringback(y_pred_argmax, stride, gg,ll):
    imageV= y_pred_argmax.shape[0]
    imageV1= y_pred_argmax.shape[1] 
    imageV2= y_pred_argmax.shape[2]
    stacked_data11 = np.empty(((gg, ll)))
    i=0
    
    for r in range(0,gg,stride):
        for c in range(0,ll,stride):
            img2=y_pred_argmax[i]
            stacked_data11[r:r+stride, c:c+stride]=img2
            i=i+1   
                
    cv2.imwrite(r"K:\Parisa\Paluxy_secondProject\data\0p34\results\ww111.tiff",stacked_data11)#cv2 always Write BGR so you should convert it. if it does not work try cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(rf"../Reattached/reatached_{im}.tiff",cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))#cv2 always Write BGR so you should convert it. if it does not work try cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
bringback(pred_value, 512,3584 ,8192)
#np.where(y_pred_argmax==y_test[:,:,:,0],1,0)

import tensorflow 
from tensorflow  import metrics
m = tensorflow.keras.metrics.Accuracy()
m.update_state(y_test, y_pred_argmax)
m.result().numpy()
#tensorflow.keras.metrics.Accuracy(y_test[:,:,:,0].reshape(-1,1), y_pred_argmax.reshape(-1,1))
tensorflow.math.confusion_matrix(y_test.reshape(-1), y_pred_argmax.reshape(-1),name=True)
from sklearn.metrics import confusion_matrix
confusion_S1 = confusion_matrix(y_test.reshape(-1), y_pred_argmax.reshape(-1),labels=np.array(np.unique(y_test.reshape(-1))))
print (confusion_S1)
pred_value=labelencoder.inverse_transform(y_pred_argmax.reshape(-1))
pred_value=pred_value.reshape(y_pred_argmax.shape[0],y_pred_argmax.shape[1],y_pred_argmax.shape[2])
################################################## IOU & F1 score


#Using built in keras function
from tensorflow.keras.metrics import MeanIoU
n_classes = 7
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(Y_test_max, y_pred_argmax)
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
Y1=Y_test_max.reshape(-1)
Y2=y_pred_argmax.reshape(-1)
sklearn.metrics.f1_score(Y1, Y2, average='macro')
# plt.imshow(train_images[0, :,:,0], cmap='gray')
# plt.imshow(train_masks[0], cmap='gray')
#######################################################################

##################################################################### SAVE HISTORY (loss, val)
#df_loss_accuraccy = pd.DataFrame([])
#df_loss_accuraccy.to_csv("df_loss_accuraccy.csv")
pd.DataFrame.from_dict(history.history).to_csv("loss_n12_Marcellus.csv")

import segmentation_models as sm
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss, categorical_focal_jaccard_loss
from segmentation_models.metrics import iou_score

BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

# load your data
#x_train, y_train, x_val, y_val = load_data(...)

# preprocess input
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)



from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model

# define number of channels
N = X_train.shape[-1]

base_model = sm.Unet(backbone_name='resnet34', encoder_weights='imagenet', classes=7, activation='softmax')

inp = Input(shape=(None, None, N))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)

model = Model(inp, out, name=base_model.name)
LR = 0.0001
optim = tensorflow.keras.optimizers.Adam(LR)
# continue with usual steps: compile, fit, etc..

# define model
#model = sm.Unet('resnet34', encoder_weights='imagenet', classes=7, activation='softmax')
#model = Unet(BACKBONE, encoder_weights='imagenet')
model.compile(optim, loss=categorical_focal_jaccard_loss, metrics=[iou_score])

# fit model
#https://github.com/qubvel/segmentation_models/blob/master/docs/tutorial.rst
history = model.fit(X_train, y_train_cat, 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=300, 
                    validation_data=(X_val, yval_cat), 
                    #class_weight=class_weights,
                    shuffle=False,
                    callbacks=[stop_early])