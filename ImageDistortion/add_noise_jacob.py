import cv2
import numpy
import shutil
import os
from random import random

def salt_pepper(img_path):
    img = cv2.imread(img_path)   # reads an image in the BGR format

    d = .2
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


