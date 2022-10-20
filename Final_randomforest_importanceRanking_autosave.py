# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:30:12 2020

@author: pza0029

Random forest with 10 features.
I used creatFeatures.py code to create dataframe of my x variables and then save them as pickle file "Documents.pkl"
#this is for storing (serialize) and loading (deserialize)
confusion matrix,
"""

###################################################################################
###################################################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle
import cv2


###################################################################################
####################################################################################

############## Load data##############################
image1 = pd.read_pickle("Documents.pkl") #dataframe of features + raw image 
labels = cv2.imread("converted_label.tiff", 0) # labels: we have it as tiff file with 7 classes

##show the image
# cv2.imshow('labels',labels)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


y =  labels.reshape(-1,1)#reshape lables to have an array 2d dimention
#y =  np.atleast_2d(y ).T  #https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html


"""append to be sure we have labels and their corresponded features at same size 
and since we shuffle them we will have the correct label in front of 
their corresponded features
"""
data = np.append(image1, y, axis=1)
#data = np.hstack((image1, y))

###################################################################################
###################################################################################

############test train split#####################
train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=0)


X_train = np.array(train[:,:-1])
Y_train = np.array(train[:,-1])
X_test = np.array(test[:,:-1])
Y_test =  np.array(test[:,-1])
#X_train = X_train.reshape(-1,1)
#X_test = X_test.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)
Y_test = Y_test.astype('str')
Y_train = Y_train.astype('str')

del train # it helps me to delete the variables that I do not need them.
del test
del data
del labels

###################################################################################
###################################################################################

###################apply the model#######################
clf = RandomForestClassifier(random_state=0, n_estimators=200, min_samples_split=10, max_depth=70, shuffle=False,n_jobs=-1)
clf.fit(X_train, Y_train)


###################################################################################
###################################################################################

##################save the trained model using pickle###
#https://www.geeksforgeeks.org/saving-a-machine-learning-model/
#https://stackabuse.com/scikit-learn-save-and-restore-models/
filename = 'finalized_model.sav'
import pickle
pickle.dump(clf , open(filename, 'wb'))

###################################################################################
###################################################################################

""" 
########################################################## 
sometime later... load the model
## uncomment this part if you want to used the saved model and comment the next part. 
if this is the first time ignore this part and go to the next part.
#########################################################

############load the model from disk#####################
filename = 'finalized_model.sav'
pickle_model = pickle.load(open(filename, 'rb'))

################get the accuracy and predicted Y##########
result = pickle_model.score(X_test, Y_test)
print(result)
Ypredict = pickle_model.predict(X_test)

########## get importance values #########################
#https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e
#https://machinelearningmastery.com/calculate-feature-importance-with-python/
importance = pickle_model.feature_importances_
########### summarize feature importance ################
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
from matplotlib import pyplot
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
pyplot.savefig('feature importance_RF10features.png')
#same as above but with the name of features
feature_importances = pd.DataFrame(pickle_model.feature_importances_, index = image1.columns,columns=['importance']).sort_values('importance',ascending=False)
feature_importances.to_csv('feature_importances.csv')

############ save the predicted results#################
predict = Ypredict.reshape(-1,1)
column_names = ['Y_predict', 'Y_test']
df1 = pd.DataFrame(np.hstack([predict, Y_test]), columns=column_names)
#df1.to_csv('Y_predict.csv',index=False)
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
res.to_csv('confusion_matrix_RF10Features.csv')
"""


###################################################################################
###################################################################################

################################################################################
####use this part if it is the first time. sometime later we can use
#the commented code above to load the model and do the modeling.
################################################################################

################get the accuracy and predicted Y#########
result = clf.score(X_test, Y_test)
print(result)
Ypredict = clf.predict(X_test)


########## get importance values #######################

#https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e
#https://machinelearningmastery.com/calculate-feature-importance-with-python/
importance = clf.feature_importances_

########### summarize feature importance ################

for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
from matplotlib import pyplot as plt
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
plt.savefig('feature importance_RF10features.png')
#same as above but with the name of features
feature_importances = pd.DataFrame(clf.feature_importances_, index = image1.columns,columns=['importance']).sort_values('importance',ascending=False)
feature_importances.to_csv('feature_importances.csv')

############ save the predicted results##################
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
res = sns.heatmap(confusion_s1_df, annot=True, fmt="d",)
res.invert_yaxis()
res.to_csv('confusion_matrix_RF10Features.csv')

###################################################################################
###################################################################################