# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:35:21 2020

@author: pza0029
"""

import os
import cv2
import glob 
import numpy as np
import matplotlib.pyplot as plt


os.chdir(r'C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\test') #main folder
folder_with_image = ''



#folder_dir_1 = os.listdir(folder_with_image)


#for i in range(len(folder_dir_1)):

#img_dir = "" # Enter Directory of all images  
data_path = os.path.join(folder_with_image, '*.tiff') 
files = glob.glob(data_path) 
#data = [] 
for f1 in files: 
    name = f1.split('.')[0].split('_')[-1]
    # read image
    #img = cv2.imread("Auburn_Shale_Lt_btCore_20X_0p8um_bin2_recon005.tiff",-1)
    img = cv2.imread(f1,0) 
    #make a vector to give it to kmean clustering.
    img_reshape = img.reshape(-1) # if it had more channel ((-1,3)) to have 3 chnnel or 3 vector for each channel.

    #kmean clustering use cv2. it is better for images.
    # convert to np.float32
    #img_reshape_float = img_reshape 
    img_reshape_float = np.float32(img_reshape)
    img_reshape_float.max()
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1)
    K = 5 #number of cluster
    attempts = 20 #
    '''attempts is : Flag to specify the number of times the algorithm is executed using different /
    initial labellings. The algorithm returns the labels that yield the best compactness. /
    This compactness is returned as output.
    '''
    ret,label,center=cv2.kmeans(img_reshape_float,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS) # or cv2.KMEANS_RANDOM_CENTERS

    #get the center
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    label_reshape = label.reshape((img.shape))
    cv2.imwrite(f"C:/Users/pza0029/Shale_project/Mancos_ziess_test/core1/K_mean_1features_results/label_reshape{name}.tif", res2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#######################relabel###############################################
img = cv2.imread(r"C:\Users\pza0029\Shale_project\Mancos_ziess_test\core1\K_mean_1features_results\label_reshaperecon359.tif",0) 


plt.figure(figsize=(15,15))
plt.imshow(img)
#plt.imsave("18.tiff",all_segments)

#make binary image segments
segm1= (img == 0)
segm2= (img == 54)
segm3= (img == 43)
segm4= (img == 49)
segm5= (img == 100)
# segm6= (img == 5)
# segm7= (img == 6)
# segm8= (img == 7)
# make an empty array same size as our image
all_segments = np.zeros((res2.shape[0],res2.shape[1]))
all_segments[segm1]=187
all_segments[segm2]=188
all_segments[segm3]=158
all_segments[segm4]=49
all_segments[segm5]=209
all_segments  = np.uint8(all_segments )
cv2.imwrite("K_mean_1dimention/relable/label_reshaperecon359.tif", all_segments)


# plt.figure(figsize=(15,15))
# plt.imshow(all_segments)
# plt.imsave("18.tiff",all_segments)


# cv2.imshow('res2',res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# cv2.imwrite("KMean_labels.tif", res2)

# img5 = cv2.imread("KMean_labels.tif",-1)

#image = img_as_ubyte(image) # it just divide your image to 256

# scale the image to 8 bit . I think it is similar the work the imageJ does
# min = img .min()
# max = img .max()
# img1 =((img  - min)/(max-min)) * 255

# #kmean clustering use sklearn
# kmeans = KMeans(n_clusters=6, init = "K-means++", n_init=10,
#        max_iter=300, tol=1e-4, precompute_distances='auto',
#        verbose=0, random_state=None, copy_x=True,
#        n_jobs=None, algorithm='auto') 

# kmeans.fit(img1)
# labels = kmeans.predict(img1)
# centroids = kmeans.cluster_centers_

# cv2.imshow("labels",labels)
# #cv2.imshow("original",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

########evaluate############################################################################################################
#evaluate. read all of them and then get the accuracy and confusion matrix
##https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import cv2
import glob 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_path = os.path.join(r"C:\Users\pza0029\Box\Shared with Parisa\CT\Marcellus_ziess_test\labled\350_360\K_mean_1dimention\relable", '*.tif') 
data_path2 = os.path.join(r"C:\Users\pza0029\Box\Shared with Parisa\CT\Marcellus_ziess_test\labled\350_360\labels", '*.tif') 
K_mean_results = glob.glob(data_path) 
labled = glob.glob(data_path2) 
Y_test = []
predict=[] 
arr = []
arr.append([1,2,3])
for f1 in range(len(labled)):
    
    #name = f1.split('.')[0].split('_')[-1]
    # read image
    #img = cv2.imread("Auburn_Shale_Lt_btCore_20X_0p8um_bin2_recon005.tiff",-1)
    img = cv2.imread(K_mean_results[f1],0) 
    img2 = cv2.imread(labled[f1],0)
    #make a vector to give it to kmean clustering.
    img_reshape = img.reshape(-1) # if it had more channel ((-1,3)) to have 3 chnnel or 3 vector for each channel.
    img_reshape2 = img2.reshape(-1)
    Y_test =np.append(Y_test , img_reshape2)
    predict=np.append(predict , img_reshape)
   
accuracy_score(Y_test, predict)
confusion_S1 = confusion_matrix(Y_test, predict, labels=[188,87,158,209,76,49])
print (confusion_S1)

# cmap = sns.cubehelix_palette(light=5, as_cmap=True)
confusion_s1_df = pd.DataFrame(confusion_S1)
confusion_s1_df.index= [188,87,158,209,76,49]
confusion_s1_df.columns= [188,87,158,209,76,49]
res = sns.heatmap(confusion_s1_df, annot=True, fmt="d",  )
res.invert_yaxis()
confusion_s1_df.to_csv('confusion_matrix_K_mean_results.csv')
