from django.shortcuts import render, redirect
from django.http import HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .forms import *
from PIL import Image
from .ImageDistortion.add_noise import salt_pepper_noise, gaussian_noise
from .ImageDistortion.add_patterns import add_random_patterns
from .ImageDistortion.add_blur import add_gaussian_blur, add_vertical_blur, add_horizontal_blur, add_box_blur
from .ImageDistortion.unsharp_masking import unsharp_mask
from .ImageDistortion.contrast import increase_contrast 
import cv2
import numpy as np
from tensorflow import keras
import shutil
import os
import sys
import time
from .tiler.tiler import Tiler

def call_alg(request):
    #input validation needed for most of these
    if request.method == 'GET':
        if request.GET['wait'] == "true":
            time.sleep(0.5)
        alg_name = request.GET['alg_name']
        print(alg_name)
        if alg_name == "addSaltAndPepperNoise":
            val = float(request.GET['val'])
            return run_alg(request, add_salt_and_pepper_noise_wrapper, val)
        elif alg_name ==  "addGaussianNoise":
            val = float(request.GET['val'])
            return run_alg(request, add_gaussian_noise_wrapper, val)
        elif alg_name == "addGaussianBlur":
            val = int(request.GET['val'])
            return run_alg(request, add_gaussian_blur_wrapper, val)
        elif alg_name == "addVerticalBlur":
            val = int(request.GET['val'])
            return run_alg(request, add_vertical_blur_wrapper, val)
        elif alg_name == "addHorizontalBlur":
            val = int(request.GET['val'])
            return run_alg(request, add_horizontal_blur_wrapper, val)
        elif alg_name == "addBoxBlur":
            val = int(request.GET['val'])
            return run_alg(request, add_box_blur_wrapper, val)
        elif alg_name == "addInpainting":
            val = float(request.GET['val'])
            return run_alg(request, add_patterns_wrapper, val)
        elif alg_name == "removeSaltAndPepperNoise":
            val = int(request.GET['val'])
            return run_alg(request, remove_salt_and_pepper_noise_wrapper, val)
        elif alg_name == "removeGaussianNoise":
            val = int(request.GET['val'])
            return run_alg(request, remove_gaussian_noise_wrapper, val)
        elif alg_name == "removeGaussianBlur":
            return run_alg(request, remove_gaussian_blur_wrapper)
        elif alg_name == "removeHorizontalBlur":
            return run_alg(request, remove_gaussian_blur_wrapper)
        elif alg_name == "removeVerticalBlur":
            return run_alg(request, remove_vertical_blur_wrapper)
        elif alg_name == "removeBoxBlur":
            return run_alg(request, remove_box_blur_wrapper)
        elif alg_name == "removeInpainting":
            val = int(request.GET['val'])
            return run_alg(request, remove_inpainting_wrapper, val)
        elif alg_name == "changeToGrayscale":
            return run_alg(request, change_to_grayscale_wrapper)
#        elif alg_name == "increaseContrast":
#            val = int(request.GET['val'])
#            return run_alg(request, increase_contrast, val)
        elif alg_name == "reset":
            user_picture = Picture.objects.get(session_id=request.session.session_key)
            path = user_picture.main_img.path
            save_as_edited_image(path, user_picture)
            return FileResponse(open(user_picture.edited_img.path, 'rb'))          
        else:
            #raise 404?
            return 

#https://docs.djangoproject.com/en/3.0/ref/csrf/
@csrf_exempt
#get crsft working later
def upload(request):
    if request.method == 'POST': 
        user_picture = Picture.objects.get(session_id=request.session.session_key)
        form = PictureForm(request.POST, request.FILES, instance=user_picture) 
        if form.is_valid(): 
            form.save()
            path = user_picture.main_img.path
            save_as_edited_image(path, user_picture)
            return FileResponse(open(user_picture.main_img.path, 'rb'))

def download(request):
    user_picture = Picture.objects.get(session_id=request.session.session_key)
    path = user_picture.edited_img.path
    #change the name of the returned file?
    head, tail = os.path.split(path)
    response = FileResponse(open(path, "rb"), as_attachment=True, filename=tail)
    return response

def run_alg(request, alg, val=None):
    user_picture = Picture.objects.get(session_id=request.session.session_key)
    img = cv2.imread(user_picture.edited_img.path)
    height, width, channels = img.shape
    img = alg(img, val)
    img = Tiler.crop(img, height, width)
    cv2.imwrite(user_picture.edited_img.path, img)
    return FileResponse(open(user_picture.edited_img.path, 'rb'))

def keras_mse_l1_loss(y_actual, y_predicted):
    kb = keras.backend
    loss = kb.mean(kb.sum(kb.square(y_actual - y_predicted))) + kb.mean(kb.sum(kb.abs(y_predicted))) * 0.004
    return loss

def apply_neural_net(img, filename):
    file_path = os.path.join(settings.STATIC_ROOT, "MainPage", "h5", filename)
    new_model = keras.models.load_model(file_path, custom_objects = {'keras_mse_l1_loss': keras_mse_l1_loss})
    tiles = Tiler.tile(img, 32, 32)
    tiles = tiles / 255
    h, w, th, tw, channels = tiles.shape
    tiles = tiles.reshape(h*w, th, tw, channels)
    out = new_model.predict(tiles)
    out = out * 255
    img = np.minimum(img, np.full(img.shape, 255))
    img = np.maximum(img, np.full(img.shape, 0))
    out = out.astype(np.uint8)
    out = out.reshape(h, w, th, tw, channels)
    composite = Tiler.stitch(out)

    return composite
       
def remove_inpainting_wrapper(img, d):
    filename = "inpainting/inpainting_" + str(d) + ".h5"
    return apply_neural_net(img, filename)

def remove_gaussian_noise_wrapper(img, d):
    filename = "gaussian_noise/gaussian_noise_" + str(d) + ".h5"
    return apply_neural_net(img, filename)

def remove_salt_and_pepper_noise_wrapper(img, d):
    filename = "salt_and_pepper_noise/salt_and_pepper_noise_" + str(d) + ".h5"
    return apply_neural_net(img, filename)

def remove_gaussian_blur_wrapper(img, d):
    filename = "gaussian_blur/gaussian_blur.h5"
    return apply_neural_net(img, filename)

def remove_horizontal_blur_wrapper(img, d):
    filename = "horizontal_blur/horizontal_blur.h5"
    return apply_neural_net(img, filename)

def remove_vertical_blur_wrapper(img, d):
    filename = "vertical_blur/vertical_blur.h5"
    return apply_neural_net(img, filename)

def remove_box_blur_wrapper(img, d):
    filename = "box_blur/box_blur.h5"
    return apply_neural_net(img, filename)

def add_patterns_wrapper(img, d):
    f = 10
    img = img / 255
    img = add_random_patterns(img, d, f, f, f)
    img = img * 255
    img = img.astype(np.uint8)
    return img


def add_salt_and_pepper_noise_wrapper(img, d):
    img = salt_pepper_noise(img, d)
    return img 

def add_gaussian_noise_wrapper(img, d):
    img = gaussian_noise(img, d)
    return img

def add_gaussian_blur_wrapper(img, d):
    img = img / 255
    img = add_gaussian_blur(img, d, 1.0)
    img = img * 255
    img = img.astype(np.uint8)
    return img

def add_vertical_blur_wrapper(img, d):
    img = img / 255
    img = add_vertical_blur(img, d)
    img = img * 255
    img = img.astype(np.uint8)
    return img

def add_horizontal_blur_wrapper(img, d):
    img = img / 255
    img = add_horizontal_blur(img, d)
    img = img * 255
    img = img.astype(np.uint8)
    return img

def add_box_blur_wrapper(img, d):
    img = img / 255
    img = add_box_blur(img, d)
    img = img * 255
    img = img.astype(np.uint8)
    return img

def change_to_grayscale_wrapper(img, val):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def index(request): 
    request.session.cycle_key()
    if not request.session.session_key:
        request.session.create()
    if not Picture.objects.filter(session_id = request.session.session_key).exists():
        user_picture = Picture.objects.create(session_id=request.session.session_key)
        path = user_picture.main_img.path
        save_as_edited_image(path, user_picture)
    return render(request, 'MainPage/index.html') 

def save_as_edited_image(img_path, user_picture):
        head, tail = os.path.split(img_path)
        edited_img_directory = head + "/edited_images/"
        shutil.copy(img_path, edited_img_directory)
        user_picture.edited_img = edited_img_directory + tail
        user_picture.save()

def testIndex(request): 
    return render(request, 'MainPage/testTemplate.html') 
