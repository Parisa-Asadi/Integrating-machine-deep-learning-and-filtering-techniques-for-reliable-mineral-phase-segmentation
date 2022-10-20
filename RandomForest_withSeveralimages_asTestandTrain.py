# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:55:50 2020

@author: pza0029
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:30:12 2020

@author: pza0029

Random forest with 1 features.
#this is for storing (serialize) and loading (deserialize)
confusion matrix,
"""

#from sklearn.svm import SVC
import numpy as np

import pandas as pd
#import sklearn 
from sklearn.model_selection import train_test_split
#from sklearn.utils import resample
#import math
from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import StratifiedShuffleSplit
#import matplotlib.pyplot as plt
#import seaborn as sn
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle
import cv2
import os
import glob 
import matplotlib.pyplot as plt

import time
start = time.time()
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
        #name = f1.split('.')[0].split('_')[-1]
            #name = f1.split('.')[0].split('_')[-1]
        # read image
        #img = cv2.imread("Auburn_Shale_Lt_btCore_20X_0p8um_bin2_recon005.tiff",-1)
        img = cv2.imread(label[f1],0) 
        img2 = cv2.imread(image[f1],0)
        #make a vector to give it to kmean clustering.
        img_reshape = img.reshape(-1,1) # if it had more channel ((-1,3)) to have 3 chnnel or 3 vector for each channel.
        img_reshape2 = img2.reshape(-1,1)
        Y =np.append(Y , img_reshape)
        X_train=np.append(X_train , img_reshape2)
    return Y, X_train
######################################################

#change directory
main_dir=r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1' #main folder
folder_with_image = 'train'
folder_with_labeledimage = 'train\labels'
Y_train, X_train =  integrated_setimage(main_dir, folder_with_image, folder_with_labeledimage )
#___for test_____
folder_with_image1 = 'test'
folder_with_labeledimage1 = 'test\labels'
Y_test, X_test =  integrated_setimage(main_dir, folder_with_image1, folder_with_labeledimage1 )
# #y =  np.atleast_2d(y ).T  #https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html
# #x = image1.reshape(-1,1)
# #x= image1

# """append to be sure we have labels and their corresponded features at same size 
# and since we shuffle them we will have the correct label in front of 
# their corresponded features
# """
# data = np.append(image1, y, axis=1)
# #data = np.hstack((image1, y))
# #test train split
# train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=0)


# X_train = np.array(train[:,:-1])
# Y_train = np.array(train[:,-1])
# X_test = np.array(test[:,:-1])
# Y_test =  np.array(test[:,-1])
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)
# Y_test = Y_test.astype('str')
# Y_train = Y_train.astype('str')

###################apply the model#######################
clf = RandomForestClassifier(random_state=0, n_estimators=200, min_samples_split=10, max_depth=70, shuffle=False,n_jobs=-1)
clf.fit(X_test, Y_test)

##################save the trained model using pickle##########################
#https://www.geeksforgeeks.org/saving-a-machine-learning-model/
#https://stackabuse.com/scikit-learn-save-and-restore-models/
filename = 'RFmodel.sav'
import pickle
pickle.dump(clf , open(filename, 'wb'))


""" 
######## some time later... load the model## uncomment this part if you want 
#to used the saved model and comment the next part
 
#load the model from disk
filename = 'RFmodel.sav'
pickle_model = pickle.load(open(filename, 'rb'))
result = pickle_model.score(X_test, Y_test)
print(result)
Ypredict = pickle_model.predict(X_test)

# get importance
#https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e
#https://machinelearningmastery.com/calculate-feature-importance-with-python/
importance = pickle_model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
from matplotlib import pyplot
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
#same as above but with the name of features
feature_importances = pd.DataFrame(pickle_model.feature_importances_, index = image1.columns,columns=['importance']).sort_values('importance',ascending=False)
feature_importances.to_csv('feature_importances.csv')
# same the predicted results
predict = Ypredict.reshape(-1,1)
column_names = ['Y_predict', 'Y_test']
df1 = pd.DataFrame(np.hstack([predict, Y_test]), columns=column_names)
#df = pd.DataFrame(Y_predict)
##df1.to_csv('Y_predict.csv',index=False)
predictions = 'predictions_RF_10features.sav'
pickle.dump(df1 , open(predictions, 'wb'))

#############confusion####################
from sklearn.metrics import confusion_matrix
confusion_S1 = confusion_matrix(Y_test, predict)
print (confusion_S1)
import seaborn as sns
# cmap = sns.cubehelix_palette(light=5, as_cmap=True)

confusion_s1_df = pd.DataFrame(confusion_S1)
res = sns.heatmap(confusion_s1_df, annot=True, fmt="d",  )
res.invert_yaxis()
res.to_csv('confusion_matrix_RF10Features.csv',index=False)
"""


####use this part if it is the first time. sometime later we can use the commented code above to load the model and do the modeling.

result = clf.score(X_test, Y_test)
print(result)
Ypredict = clf.predict(X_test)
print(f'Time: {time.time() - start}')
# get importance
#https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e
#https://machinelearningmastery.com/calculate-feature-importance-with-python/
importance = clf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
from matplotlib import pyplot
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
#same as above but with the name of features
feature_importances = pd.DataFrame(clf.feature_importances_, index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)
feature_importances.to_csv('FNN&RF/results_RF_1features/feature_importances.csv')
# same the predicted results
predict = Ypredict.reshape(-1,1)
column_names = ['Y_predict', 'Y_test']
df1 = pd.DataFrame(np.hstack([predict, Y_test]), columns=column_names)
#df = pd.DataFrame(Y_predict)
##df1.to_csv('Y_predict.csv',index=False)

predictions = 'predictions_RF_1features.sav'
pickle.dump(df1 , open(predictions, 'wb'))

#############confusion####################
from sklearn.metrics import confusion_matrix
confusion_S1 = confusion_matrix(Y_test, Ypredict,labels=[158, 187, 188, 209, 76])
print (confusion_S1)
import seaborn as sns
# cmap = sns.cubehelix_palette(light=5, as_cmap=True)

confusion_s1_df = pd.DataFrame(confusion_S1)
# confusion_s1_df.index= ['158.0', '187.0', '188.0', '209.0', '76.0']
# confusion_s1_df.columns= ['158.0', '187.0', '188.0', '209.0', '76.0']
res = sns.heatmap(confusion_s1_df, annot=True, fmt="d",  )
#res.invert_yaxis()
confusion_s1_df.to_csv(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\FNN&RF\results_RF_1features\confusion_matrix_RF1Features_test_withoutname.csv')

####recreate the segmented images with RF###########
os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\FNN&RF\results_RF_1features') #main folder
Num=10 #number of images 
NumArray = int(len(Ypredict)/Num)
for i in range(0,Num):
    Ypredict1= Ypredict[NumArray*i:(i+1)*NumArray]
    final= np.uint8(Ypredict1.reshape(994,968))
    #cv2.imwrite(f"K_mean_1dimention/label_reshape{name}.tif", res2)
    cv2.imwrite(f"RF_segmentedimage_{i}.tif",final)
