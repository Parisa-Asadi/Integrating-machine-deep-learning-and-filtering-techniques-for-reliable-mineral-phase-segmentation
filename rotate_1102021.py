# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:59:56 2021

@author: pza0029
"""

from PIL import Image 
 
  
img = Image.open(r"C:\Users\pza0029\Box\Ph.D Civil\My_total_works\work\Paluxy_resolution_ML\croppedat256\inputs\si.tif") 
im11g = Image.open(r"C:\Users\pza0029\Box\Ph.D Civil\My_total_works\work\Paluxy_resolution_ML\croppedat256\BSE.tif")  
rotate_img= img.rotate(-0.5)
rotate_img.show() 
rotate_img.save(r"0p5\0p5.tif")
img.close()
print("hi")
########