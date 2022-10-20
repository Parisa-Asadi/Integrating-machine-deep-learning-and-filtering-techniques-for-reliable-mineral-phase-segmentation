# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:59:56 2021
#parisa Asadi
to resize
rotate a set of images
and crop

@author: pza0029
"""
######import library
from PIL import Image 
import os

#change path
path = r"C:\Users\pza0029\Box\Ph.D Civil\My_total_works\work\Paluxy_resolution_ML\old\WithoutRotation\croppedat256\inputs\0p6"
os.chdir(path) 
###### Store the image file names in a list as long as they are jpgs
images = [f for f in os.listdir(path) if (os.path.splitext(f)[-1] == '.tif')]

###### make new directory to save rotated files
# os.makedirs("0p5")
# os.makedirs("0p6")
#os.makedirs("crop")


###### rotate and save
for i, image in enumerate(images):
    img = Image.open(image)
    width, height = img.size

    # ####resize
    # img = img.resize((400, 400)) #change
    
    # #### crop
    
    # Setting the points for cropped image
    # left = 18
    # top = 42
    # right = 8476
    # bottom = 4010
    # if height==364:
    #     top = 65
    #     bottom = 270
    # Cropped image of above dimension
    # img = img.crop((left, top, right, bottom))
    ####
    rotate_img= img.rotate(-0.5) #change
    rotate_img1= img.rotate(-0.6) #change
    rotate_img.show() 
    rotate_img.save(rf"0p5\0p5_{images[i]}.tif")
    rotate_img1.save(rf"0p6\0p6_{images[i]}.tif")
    # img.save(rf"crop\{images[i]}.tif")
    img.close()
print("hi")
########






###### cropped and save
for i, image in enumerate(images):
    img = Image.open(image)
    width, height = img.size

    # ####resize
    # img = img.resize((400, 400)) #change
    # #### crop
    
    # Setting the points for cropped image
    left = 18
    top = 42
    right = 8476
    bottom = 4010
    # if height==364:
    #     top = 65
    #     bottom = 270
    # Cropped image of above dimension
    img = img.crop((left, top, right, bottom))
    ####
    # rotate_img= img.rotate(-0.5) #change
    # rotate_img1= img.rotate(-0.6) #change
    #rotate_img.show() 
    # rotate_img.save(rf"0p5\0p5_{images[i]}.tif")
    # rotate_img1.save(rf"0p6\0p6_{images[i]}.tif")
    img.save(rf"crop\{images[i]}.tif")
    img.close()
print("hi")
########






###### cropped, rotate and save
for i, image in enumerate(images):
    img = Image.open(image)
    width, height = img.size

    # ####resize
    # img = img.resize((400, 400)) #change
    # #### crop
    
    # Setting the points for cropped image
    left = 18
    top = 42
    right = 8476
    bottom = 4010
    # if height==364:
    #     top = 65
    #     bottom = 270
    # Cropped image of above dimension
    img = img.crop((left, top, right, bottom))
    ####
    rotate_img= img.rotate(-0.5) #change
    rotate_img1= img.rotate(-0.6) #change
    rotate_img.show() 
    rotate_img.save(rf"0p5\0p5_{images[i]}.tif")
    rotate_img1.save(rf"0p6\0p6_{images[i]}.tif")
    # img.save(rf"crop\{images[i]}.tif")
    img.close()
print("hi")
########