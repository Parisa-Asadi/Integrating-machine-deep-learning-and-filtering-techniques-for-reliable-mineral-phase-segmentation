# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:24:19 2021
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/
@author: Parisa
"""

import pandas as pd

actual = pd.read_csv(r"F:\Parisa\data_of_era5_for_2000_research\ERA5\California\final_ERA5_california\YtestCART4.csv")
predicted = pd.read_csv(r"F:\Parisa\data_of_era5_for_2000_research\ERA5\California\final_ERA5_california\predCART4.csv")

# confusion matrix in sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# confusion matrix

matrix = confusion_matrix(actual,predicted, labels=[0,1])
print('Confusion matrix : \n',matrix)

# outcome values order in sklearn
tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[0,1]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(actual,predicted,labels=[0,1])
print('Classification report : \n',matrix)