import cv2
import numpy as np
import shutil
import os
from random import random

#d should be between 0 and 1, determines level of salt/pepper noise
def salt_pepper(img_path, d):
    img = cv2.imread(img_path)  

    height = img.shape[0]
    width = img.shape[1]
    for y in range (0, height):
        for x in range (0, width):

            r = random()
            if (r <= d/2):
                img [y, x, 0] = 255
                img [y, x, 1] = 255
                img [y, x, 2] = 255
            elif(r <= d):
                img [y, x, 0] = 0
                img [y, x, 1] = 0
                img [y, x, 2] = 0
            
    cv2.imwrite(img_path,img)


#for significant amount of noise, sigma should be above 10.
def gaussian(img_path, sigma):
    img = cv2.imread(img_path)
    
    height = img.shape[0]
    width = img.shape[1]

    
    s = np.random.normal(0, sigma, height*width*3)
    ind = 0
    for y in range (0, height):
        for x in range (0, width):
            for z in range (0, 3):
               value = img [y, x, z] + s[ind] 
               value = min(value,255)
               value = max(0, value)
               img[y, x, z] = value
               ind += 1
    
    cv2.imwrite(img_path,img) 

