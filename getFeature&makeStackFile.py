# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:25:07 2020

@author: pza0029
"""
    ############################################
    #####--------get libraries--------------####
    ############################################
import os
import cv2
import glob 
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage.filters import sobel, sobel_h, sobel_v
#https://stackoverflow.com/questions/33686005/trouble-importing-filters-using-skimage
#pip install -U scikit-image #use it in anoconda prompt then restart kernel 
from skimage.filters import difference_of_gaussians
from skimage.filters import hessian
from skimage.filters import gabor, laplace
import pandas as pd
import pickle 
############################################
#####--------change directory-----------####
############################################
os.chdir(r'C:\My_works\Mancos&Marcellus\Marcellus\Train') #main folder
folder_with_image = ''
df = pd.DataFrame()
#folder_dir_1 = os.listdir(folder_with_image)
#for i in range(len(folder_dir_1)):
#img_dir = "" # Enter Directory of all images
############################################
#####--------read images----------------####
############################################  
data_path = os.path.join(folder_with_image, '*.tiff') 
files = glob.glob(data_path) 

# img = cv2.imread(files[39],0)
# img = np.flipud(img)
# plt.figure(figsize=(10,10))
# plt.imshow(img)
# plt.show()

#data = [] 
for  count, f1 in enumerate(files): 
    df = pd.DataFrame()
    name = f1.split('.')[0].split('_')[-1]
    # read image
    #img = cv2.imread("Auburn_Shale_Lt_btCore_20X_0p8um_bin2_recon005.tiff",-1)
    
    img = cv2.imread(f1,0)
    # if count<40:
    #     img = np.flipud(img) #check if your image and labels are the same otherwise flip it
    ############################################
    #####--------get filters----------------####
    ############################################
    gaussian_img = nd.gaussian_filter(img, sigma=1)
    sobel_img = sobel(img)
    sobel_h1 = sobel_h(img)
    sobel_v1 = sobel_v(img)
    differenceOfGaussians_1_10 = difference_of_gaussians(img, 1,10)
    differenceOfGaussians_1_5 = difference_of_gaussians(img, 1,5)
    differenceOfGaussians_0_5 = difference_of_gaussians(img, 0,5)
    differenceOfGaussians_1_2 = difference_of_gaussians(img, 1,2)
    hessian1= hessian(img,sigmas=1)
    hessian3= hessian(img,sigmas=5)
    differenceOfHessians = hessian3 - hessian1
    gaborfilt_realx, gaborfilt_imagx = gabor(img, theta = 0, frequency=0.7,mode='nearest')
    gaborfilt_realy, gaborfilt_imagy = gabor(img, theta = 90, frequency=0.7,mode='nearest')
    median_blur = cv2.medianBlur(img,1)
    laplace_filter = laplace(img, ksize=3, mask=None)
    ############################################
    #####--------write_filters----------------####
    ############################################
    nameOfFilters = [gaussian_img,sobel_img,sobel_h1,sobel_v1,differenceOfGaussians_1_10,differenceOfGaussians_0_5,differenceOfGaussians_1_5,differenceOfGaussians_1_2,differenceOfHessians,gaborfilt_imagx,gaborfilt_imagy,median_blur,laplace_filter]
    nameOfFilters_string = ["gaussian_img","sobel_img","sobel_h1","sobel_v1","differenceOfGaussians_1_10","differenceOfGaussians_0_5","differenceOfGaussians_1_5","differenceOfGaussians_1_2","differenceOfHessians","gaborfilt_imagx","gaborfilt_imagy","median_blur","laplace_filter"]

    for i, nameOfFilter in enumerate(nameOfFilters):
        cv2.imwrite(f"{nameOfFilters_string[i]}_{name}.tif", nameOfFilter)
    ############################################
    ##---reshape_filters & make dataframe-----##
    ############################################
    gaussian_img = gaussian_img.reshape(-1)
    df['Gaussian s1'] = gaussian_img
    sobel_img = sobel_img.reshape(-1)
    sobel_h1 = sobel_h1.reshape(-1)
    sobel_v1 = sobel_v1.reshape(-1)
    df['sobel_img'] = sobel_img
    df['sobel_v1'] = sobel_v1
    df['sobel_h1'] = sobel_h1
    img2 = img.reshape(-1)
    df['Original Image'] = img2
    differenceOfHessians = differenceOfHessians.reshape(-1)
    df['differenceOfHessians'] = differenceOfHessians
    differenceOfGaussians_1_10 = differenceOfGaussians_1_10.reshape(-1)
    df['differenceOfGaussians_1_10'] = differenceOfGaussians_1_10
    differenceOfGaussians_1_5 = differenceOfGaussians_1_5.reshape(-1)
    df['differenceOfGaussians_1_5'] = differenceOfGaussians_1_5
    differenceOfGaussians_1_2 = differenceOfGaussians_1_2.reshape(-1)
    df['differenceOfGaussians_1_2'] = differenceOfGaussians_1_2
    differenceOfGaussians_0_5 = differenceOfGaussians_0_5.reshape(-1)
    df['differenceOfGaussians_0_5'] = differenceOfGaussians_0_5
    gaborfilt_imagx = gaborfilt_imagx.reshape(-1)
    df['gaborfilt_imagx'] = gaborfilt_imagx
    gaborfilt_imagy = gaborfilt_imagy.reshape(-1)
    df['gaborfilt_imagy'] = gaborfilt_imagy
    median_blur = median_blur.reshape(-1)
    df['median_blur'] = median_blur
    laplace_filter = laplace_filter.reshape(-1)
    df['laplace_filter'] = laplace_filter
    
    ############################################
    ##---save dataframe-----##
    ############################################
      ##############how to save it as dataframe and load it agian   
    #The easiest way is to pickle it using to_pickle:
    df.to_pickle(f"Features_{name}.pkl")  # where to save it, usually as a .pkl
    
    
############################################
##---read dataframes and get stack one -----##
############################################
# ee= cv2.imread("sobel_img_recon108.tif",-1)
# plt.figure(figsize=(10,10))
# plt.imshow(ee)
# plt.title('ee')
# plt.show()
path = r'C:\My_works\Mancos&Marcellus\Marcellus\Train'
os.chdir(r'C:\My_works\Mancos&Marcellus\Marcellus\Train') #main folder
# Store the image file names in a list as long as they are jpgs
images = [f for f in os.listdir(path) if (os.path.splitext(f)[-1] == '.pkl')]

#arr_feature = np.empty((0,9), int)
#https://www.kite.com/python/answers/how-to-create-an-empty-dataframe-with-column-names-in-python
#https://stackoverflow.com/questions/29351840/stack-two-pandas-data-frames/29351948
arr_feature =  pd.DataFrame()
for i, image in enumerate(images):
    #Then you can load it back using:
    df = pd.read_pickle(image)
    print(f'the {i} file added')
    #arr_feature= np.vstack(arr_feature,df)
    arr_feature=pd.concat([arr_feature,df],ignore_index=True)
    print(f'the {i} file added1')
# df.head(2)
arr_feature.to_pickle("Train_feature.pkl") 
print("hi")

#Train_feature = pd.read_pickle("Train_feature.pkl")

############################################
##---read labels and get stack one -----##
############################################
# ee= cv2.imread("sobel_img_recon108.tif",-1)
# plt.figure(figsize=(10,10))
# plt.imshow(ee)
# plt.title('ee')
# plt.show()
path = r'C:\My_works\Mancos&Marcellus\Marcellus\Train\labels'
os.chdir(r'C:\My_works\Mancos&Marcellus\Marcellus\Train\labels') #main folder
#path = './labels'  # relative path NOTE: ../  goes one folder back into your directory
# Store the image file names in a list as long as they are jpgs
images = [f for f in os.listdir(path) if (os.path.splitext(f)[-1] == '.tif')]

##arr_feature = np.empty((0,9), int)
##https://www.kite.com/python/answers/how-to-create-an-empty-dataframe-with-column-names-in-python
##https://stackoverflow.com/questions/29351840/stack-two-pandas-data-frames/29351948
arr_label= np.array([[]], dtype='uint8').T
for i, IM in enumerate(images):
    image = os.path.join(path, IM)
    #Then you can load it back using:
    df = cv2.imread(image,-1)
    df = df.reshape(-1,1)
    print(f'the {i} file added')
    #arr_feature= np.vstack(arr_feature,df)
    arr_label=np.concatenate((arr_label,df), axis=0)
    print(f'the {i} file added1')
# df.head(2)
#save it as .plk
#https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-a-numpy-array-and-specify-the-index-column-and-column-headers/
arr_label1= pd.DataFrame(arr_label, columns= ["labels"])
path_save_load = os.path.join(path, "Train_labels.pkl")
arr_label1.to_pickle(path_save_load)
print("hi")

#load it later or use it in RF or FNN
Train_labels = pd.read_pickle(path_save_load)

#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dtypes.html