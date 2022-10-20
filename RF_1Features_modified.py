# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:12:45 2021

@author: pza0029
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:40:41 2020

@author: pza0029
"""

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
import time

start = time.time()

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
import pickle
import seaborn as sns
# get files
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

path = r"K:\New folder\New_05222021\Marcellus" #K:\New folder\New_05222021\Marcellus  #C:\My_works\Mancos&Marcellus\Mancos
os.chdir(path)
Y_test = pd.read_pickle("Test_labels.pkl")
X_test = pd.read_pickle("Test_feature.pkl")
Y_train  = pd.read_pickle("Train_labels.pkl")
X_train = pd.read_pickle("Train_feature.pkl")
X_test = X_test['Original Image']
X_train = X_train['Original Image']
#https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
# Feature Scaling
# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
# Y_train = Y_train.reshape(-1,1)
# Y_test = Y_test.reshape(-1,1)
# # Y_test = Y_test.astype('str')
# Y_train = Y_train.astype('str')


#os.chdir(r"K:\New folder\New_05222021\Marcellus") #K:\New folder\New_05222021\Marcellus #C:\My_works\Mancos&Marcellus\Mancos\RF1
###################apply the model#######################
clf = RandomForestClassifier(random_state=0, n_estimators=200, min_samples_split=10,n_jobs=5, max_depth=70,class_weight="balanced")
clf.fit(X_train, Y_train)

##################save the trained model using pickle##########################
#https://www.geeksforgeeks.org/saving-a-machine-learning-model/
#https://stackabuse.com/scikit-learn-save-and-restore-models/
os.mkdir("RF1")
filename = 'RF1\RFmodel_1features.sav' #C:\My_works\Mancos&Marcellus\Mancos\RF14\RF_14feature_reults
import pickle
pickle.dump(clf , open(filename, 'wb'))


""" 
######## some time later... load the model## uncomment this part if you want 
#to used the saved model and comment the next part
 
#load the model from disk
filename = 'RFmodel_1features.sav'
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
res = sns.heatmap(confusion_s1_df, annot=True, fmt="d",  )♣
res.invert_yaxis()
res.to_csv('confusion_matrix_RF10Features.csv',index=False)
"""


####use this part if it is the↨ first time. sometime later we can use the commented code above to load the model and do the modeling.

##############################################scores confusion f_scores
result = clf.score(X_test, Y_test)
print(result)
Ypredict = clf.predict(X_test)
'''
# To calculate accuracy f-score ... for train values 
###########
result1 = clf.score(X_train, Y_train)
print(result1)
Ypredict5 = clf.predict(X_train)

#from sklearn import metrics

#################https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(Y_train,Ypredict5))
print(classification_report(Y_train,Ypredict5))
print(accuracy_score(Y_train, Ypredict5))
##############
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y1 = labelencoder.fit_transform(Y_train)
Y2=labelencoder.fit_transform(Ypredict5)
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
#############

# path = r"C:\Project_MachineLearning\Marcellus_ziess_test\labled\RF&FFNN\test\RF_14feature_reults"
# os.chdir(path)

############################################### get importance
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
feature_importances.to_csv('feature_importances_14features.csv')
# same the predicted results
#predict = Ypredict.reshape(-1,1)
column_names = ['Y_predict']
df1 = pd.DataFrame(Ypredict, columns=column_names)
#df = pd.DataFrame(Y_predict)
##df1.to_csv('Y_predict.csv',index=False)
predictions = 'predictions_RF_14features.sav'
pickle.dump(df1 , open(predictions, 'wb'))
#https://mljar.com/blog/feature-importance-in-random-forest/
# from sklearn.inspection import permutation_importance
# perm_importance = permutation_importance(clf, X_test, Y_test)
# perm_importance = pd.DataFrame(perm_importance)
# perm_importance .to_csv('feature_importances_14features.csv')
######################################################
from sklearn.inspection import permutation_importance
result12 = permutation_importance(clf, X_test, Y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result12.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result12.importances[sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()
#column_names = ['Y_predict']
df2 = pd.DataFrame(result12)
#predictions12 = 'permutationimportance_RF14.sav'
#pickle.dump(df2 , open(predictions12, 'wb'))
import pickle
pickle.dump(result12 , open(predictions12, 'wb'))

#############confusion####################
from sklearn.metrics import confusion_matrix
confusion_S1 = confusion_matrix(Y_test, Ypredict,labels=[158, 187, 188, 209, 76])
print (confusion_S1)

import seaborn as sns
# cmap = sns.cubehelix_palette(light=5, as_cmap=True)

confusion_s1_df = pd.DataFrame(confusion_S1)
# confusion_s1_df.index= ['158.0', '187.0', '188.0', '209.0', '76.0']
# confusion_s1_df.columns= ['158.0', '187.0', '188.0', '209.0', '76.0']
res = sns.heatmap(confusion_s1_df, annot=True, fmt="d" )
#res.invert_yaxis()
confusion_s1_df.to_csv('confusion_matrix_RF1Features_test_withoutname.csv')

#########################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y1 = labelencoder.fit_transform(Y_test)
Y2=labelencoder.fit_transform(Ypredict)
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


####recreate the segmented images with RF###########
os.makedirs("RF_1feature_reults")
os.chdir(r'RF_1feature_reults') #main folder
Num=21 #number of images 
NumArray = int(len(Ypredict)/Num)
for i in range(0,Num):
    Ypredict1= Ypredict[NumArray*i:(i+1)*NumArray]
    final= np.uint8(Ypredict1.reshape(994,969))#994,968 for mancos. 994,969 foe marcellus
    #cv2.imwrite(f"K_mean_1dimention/label_reshape{name}.tif", res2)
    cv2.imwrite(f"RF1features_segmentedimage_{i}.tif",final)


