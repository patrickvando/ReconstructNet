from django.shortcuts import render, redirect
from django.http import HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import *
from PIL import Image
from .ImageDistortion.add_noise import salt_pepper_noise, gaussian_noise
from .ImageDistortion.add_patterns import add_random_patterns
from .ImageDistortion.unsharp_masking import unsharp_mask
from .ImageDistortion.contrast import increase_contrast 
import cv2
import numpy
import shutil
import os
import sys
import time


def call_alg(request):
    #input validation needed for most of these
    if request.method == 'GET':
        if request.GET['wait'] == "true":
            time.sleep(0.5)
        alg_name = request.GET['alg_name']
        if alg_name == "addSaltAndPepperNoise":
            val = float(request.GET['val'])
            return run_alg(request, salt_pepper_noise, val)
        elif alg_name ==  "addGaussianNoise":
            val = float(request.GET['val'])
            return run_alg(request, gaussian_noise, val)
        elif alg_name == "addGaussianBlur":
            pass
        elif alg_name == "addVerticalBlur":
            pass
        elif alg_name == "addHorizontalBlur":
            pass
        elif alg_name == "inpainting":
            pass
        elif alg_name == "removeSaltAndPepperNoise":
            pass
        elif alg_name == "removeGaussianNoise":
            pass
        elif alg_name == "removeGaussianBlur":
            pass
        elif alg_name == "removeHorizontalBlur":
            pass
        elif alg_name == "removeVerticalBlur":
            pass
        elif alg_name == "removeInpainting":
            pass
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

def run_alg(request, alg, val):
    user_picture = Picture.objects.get(session_id=request.session.session_key)
    img = cv2.imread(user_picture.edited_img.path)
    img = alg(img, val)
    cv2.imwrite(user_picture.edited_img.path, img)
    return FileResponse(open(user_picture.edited_img.path, 'rb'))


def grayscale(img_path):
    img = cv2.imread(img_path)   # reads an image in the BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path,gray)

def contrast_button(request):
    if request.method == 'GET':
        user_picture = Picture.objects.get(session_id=request.session.session_key)
        img = Image.open(user_picture.edited_img.path)
        img = img.convert("RGB") 
        img = increase_contrast(img, 5)
        img.save(user_picture.edited_img.path)

        return FileResponse(open(user_picture.edited_img.path, 'rb'))

def sharpen_button(request):
    if request.method == 'GET':
        user_picture = Picture.objects.get(session_id=request.session.session_key)
        img = Image.open(user_picture.edited_img.path)
        img = img.convert("RGB") 
        img = unsharp_mask(img, 7, 5, 5)
        img.save(user_picture.edited_img.path)
        return FileResponse(open(user_picture.edited_img.path, 'rb'))

def gaussian_noise_button(request):
    if request.method == 'GET':
        sigma = float(request.GET['sigma'])
        user_picture = Picture.objects.get(session_id=request.session.session_key) 
        img = cv2.imread(user_picture.edited_img.path)
        img = gaussian_noise(img, sigma)
        cv2.imwrite(user_picture.edited_img.path, img)
        return FileResponse(open(user_picture.edited_img.path, 'rb'))

def salt_pepper_noise_button(request):
    if request.method == 'GET':
        d = float(request.GET['d'])
        print(d)
        user_picture = Picture.objects.get(session_id=request.session.session_key)
        img = cv2.imread(user_picture.edited_img.path)
        img = salt_pepper_noise(img, d)
        cv2.imwrite(user_picture.edited_img.path, img)
        return FileResponse(open(user_picture.edited_img.path, 'rb'))


def add_patterns_button(request):
    if request.method == 'GET':
        print(request.GET)
        max_ratio = float(request.GET['maxRatio'])
        max_lines = int(request.GET['maxLines'])
        max_circles = int(request.GET['maxCircles'])
        max_ellipses = int(request.GET['maxEllipses'])
        user_picture = Picture.objects.get(session_id=request.session.session_key)
        img = cv2.imread(user_picture.edited_img.path)
        img = add_random_patterns(img, max_ratio, max_lines, max_circles, max_ellipses)
        cv2.imwrite(user_picture.edited_img.path, img)
        return FileResponse(open(user_picture.edited_img.path, 'rb'))

def grayscale_button(request):
    if request.method == 'GET':
        user_picture = Picture.objects.get(session_id=request.session.session_key)
        grayscale(user_picture.edited_img.path)
        return FileResponse(open(user_picture.edited_img.path, 'rb'))


def reset(request):
    if request.method == 'GET':
        user_picture = Picture.objects.get(session_id=request.session.session_key)
        path = user_picture.main_img.path
        save_as_edited_image(path, user_picture)
        return FileResponse(open(user_picture.edited_img.path, 'rb'))

def index(request): 
    request.session.cycle_key()
    if not request.session.session_key:
        request.session.create()
    #assign a new database entry to the user if none exists
    if not Picture.objects.filter(session_id = request.session.session_key).exists():
        user_picture = Picture.objects.create(session_id=request.session.session_key)
        path = user_picture.main_img.path
        save_as_edited_image(path, user_picture)
    return render(request, 'MainPage/index.html') 

#user_picture is the database entry
#takes a path to an image, and saves that image as the "edited image field" in the Picture model
def save_as_edited_image(img_path, user_picture):
        head, tail = os.path.split(img_path)
        edited_img_directory = head + "/edited_images/"
        shutil.copy(img_path, edited_img_directory)
        user_picture.edited_img = edited_img_directory + tail
        user_picture.save()
 
def testIndex(request): 
    return render(request, 'MainPage/testTemplate.html') 
