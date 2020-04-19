import cv2
import numpy as np
import shutil
import os
from random import random


def salt_pepper_noise(img, d):
    height, width, channels = img.shape
    random = np.random.rand(height*width)
    random = np.tile(random, (channels, 1))
    random = np.swapaxes(random, 0, 1)
    random = random.reshape(img.shape)
    img[random < d/2] = 0
    img[random >= 1-d/2] = 255 
    return img


def gaussian_noise(img, sigma):
    height, width, channels = img.shape
    noise = np.random.normal(0, sigma, height*width*channels)
    noise = noise.reshape(height, width, channels)
    img = img + noise
    img = np.minimum(img, np.full(img.shape, 255))
    img = np.maximum(img, np.full(img.shape, 0))
    img = img.astype(np.uint8)
    return img


def add_salt_pepper_noise(img, d):
    height, width, channels = img.shape
    random = np.random.rand(height*width)
    random = np.tile(random, (channels, 1))
    random = np.swapaxes(random, 0, 1)
    random = random.reshape(img.shape)
    img[random < d/2] = 0.0
    img[random >= 1-d/2] = 1.0
    return img


def add_gaussian_noise(img, sigma):
    height, width, channels = img.shape
    noise = np.random.normal(0, sigma, (height, width, channels))
    img = img + noise
    img = np.minimum(img, np.full(img.shape, 1.0))
    img = np.maximum(img, np.full(img.shape, 0.0))
    return img
