# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 08:48:16 2021

@author: ayomy
"""

import os
import glob as glob
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from macest.classification import models as clmod
from macest.classification import utils as clut
from macest.classification import plots as clplot
from macest.classification.metrics import expected_calibration_error
from macest.model_selection import KFoldConfidenceSplit






scaler = StandardScaler()
train_files = glob.glob(os.path.join(r"C:\Users\pza0029\Box\AU_GE Data Science collaboration 2021\files\Label\CSV\New_Labels\Scenarios\Scenario_3\train","*.csv"))
data_train = pd.concat((pd.read_csv(i).assign(file=i) for i in train_files),ignore_index=True)
Conf_Train_files = glob.glob(os.path.join(r"C:\Users\pza0029\Box\AU_GE Data Science collaboration 2021\files\Label\CSV\New_Labels\Scenarios\Scenario_3\Conf_Train","*.csv"))
data_Conf_Train = pd.concat((pd.read_csv(i).assign(file=i) for i in train_files),ignore_index=True)
val_files = glob.glob(os.path.join(r"C:\Users\pza0029\Box\AU_GE Data Science collaboration 2021\files\Label\CSV\New_Labels\Scenarios\Scenario_3\val","*.csv"))
data_val = pd.concat((pd.read_csv(i).assign(file=i) for i in train_files),ignore_index=True)
test_files = glob.glob(os.path.join(r"C:\Users\pza0029\Box\AU_GE Data Science collaboration 2021\files\Label\CSV\New_Labels\Scenarios\Scenario_3\test","*.csv"))
data_test = pd.concat((pd.read_csv(i).assign(file= i) for i in test_files),ignore_index=True)


#Train set
X_train = data_train["HN1K12"].values
X_train = np.array(X_train)
X_train = X_train.reshape(-1,1)
X_train = scaler.fit_transform(X_train)

y_train = data_train[["Label"]]
h = LabelEncoder()           
y_train = h.fit_transform(y_train) 


#Conf_Train set
X_Conf_Train = data_Conf_Train["HN1K12"].values
X_Conf_Train = np.array(X_Conf_Train)
X_Conf_Train  = X_Conf_Train.reshape(-1,1)
X_Conf_Train  = scaler.fit_transform(X_Conf_Train)

y_Conf_Train = data_Conf_Train[["Label"]]
h = LabelEncoder()           
y_Conf_Train = h.fit_transform(y_Conf_Train) 


#val set
X_val = data_val["HN1K12"].values
X_val = np.array(X_val)
X_val = X_val.reshape(-1,1)
X_val = scaler.fit_transform(X_val)

y_val = data_train[["Label"]]
h = LabelEncoder()           
y_val = h.fit_transform(y_val) 


#Test set
X_test = data_test["HN1K12"].values
X_test = np.array(X_test)
X_test = X_test.reshape(-1,1)
X_test = scaler.fit_transform(X_test)

y_test_c = data_test[["Label"]]
y_test = data_test[["Label"]]
h = LabelEncoder()           
y_test = h.fit_transform(y_test) 

# Prediction model
model = KNeighborsClassifier(n_neighbors=45)
model.fit(X_train,y_train)  
prediction = model.predict(X_test.reshape(-1,1))

#encoder
def encode_part1(y_values):

    final_y = y_values
    final_y = final_y.tolist()
    final_y_predict = []
    for i in final_y: 
         if i == 2 :
            final_y_predict.append("Other")
         elif i == 1 :
            final_y_predict.append("Idle")
         elif i == 0 :
            final_y_predict.append("Cruise")
         else:
            final_y_predict.append("Taxi")
    return final_y_predict

prediction1 = encode_part1(prediction)


np.savetxt("KNN_predicted_label1_scenario_2.csv",prediction1,header = "KNN_predicted_label1",fmt ="%s", delimiter =",")
np.savetxt("KNN_y_test_scenario2.csv",y_test_c,header = "y_test_label1",fmt ="%s", delimiter =",")


#Confusion Matrix
def Confusionmatrix(y_pred,y_test):
    plt.figure(figsize=(5,5))
    ax = plt.subplot()
    cm = confusion_matrix(y_test,y_pred,labels = ["Idle","Taxi","Cruise","Other"])
    sns.heatmap(cm,annot=True ,fmt ='g',ax =ax)
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.xaxis.set_ticklabels(["Idle","Taxi","Cruise","Other"]);ax.yaxis.set_ticklabels(["Idle","Taxi","Cruise","Other"])
    ax.set_title('Confusion Matrix for KNN Model')
    return cm,ax
Confusionmatrix(prediction1, y_test_c)

# precision, recall and f1-score for each classes
print(classification_report(y_test_c,prediction1))

macest_model = clmod.ModelWithConfidence(model,X_Conf_Train,y_Conf_Train)#2

macest_model.fit(X_val,y_val)#3
conf_preds = macest_model.predict_confidence_of_point_prediction(X_test)#4


np.savetxt("KNNconfidencetest1_scenario_2_score.csv",conf_preds,header = "point_confidencetest1_score",fmt ="%s", delimiter =",")