# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 11:41:54 2021
read images in several groups from a path and process and write them in the same directory
@author: pza0029
"""
import os
os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\Train\augmented1\cropped') #main folder
#####
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


#Capture training image info as a list
Ngroup=20 #number of group you want to use to devide the images in the path to it
name= []
#GET THE name of all images in a path
for directory_path in glob.glob(r"C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\Train\augmented1\cropped"):
     for count, value in enumerate(glob.glob(os.path.join(directory_path, "*.tif"))):
         name.append(value)
#read images in "Ngroup" group and process them and write them to aviod memory shoratge
for i in range(0,len(name)/Ngroup):
    aa=50*(i)
    bb=50*(i+1) 
    if (i ==  len(name)/Ngroup): bb=50*(i+1)+1
    
    images= []
    for j in range(aa,bb):
        img = cv2.imread(name[0],cv2.IMREAD_GRAYSCALE)       
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        images.append(img)
    
    #time to bring the model you want to get results for it and then we save the image_results
    
    #save the image_results
    images = np.array(images).reshape(images.shape[0], 1, 14)
    
    ########
    from tensorflow.python.keras import losses
    model1.load_weights(r"saved_model3_FNN14\my_modelMarcellus-weights")   
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
    
    # test_loss1, test_acc1 = model1.evaluate(X_test,  Y_test, verbose=2)
    # print('\nTest accuracy:', test_acc1)
    
    #predict
    predictions2 = np.argmax(probability_model1.predict(images),axis=2)
    

#save
    os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\U-net\images_for_dataAugmentation\Marcellus\Test\Y_pred_UNet_metrics1&focal&dice_loss\color') #main folder
 
    for ii in range(predictions2.shape[0]):
        final= predictions2[ii]
        #cv2.imwrite(f"K_mean_1dimention/label_reshape{name}.tif", res2)
        segm1 = (final == 0)
        segm2 = (final == 1)
        segm3 = (final == 2)
        segm4 = (final ==3)
        segm5 = (final ==4)
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
        plt.imsave(f"U_net_{i}_{ii}.tiff",all_segments)