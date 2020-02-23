from django.shortcuts import render, redirect
from django.http import HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import *
import cv2
import numpy
import shutil
import os

def grayscale(img_path):
    img = cv2.imread(img_path)   # reads an image in the BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path,gray)

def grayscale_button(request):
    if request.method == 'GET':
        user_picture = Picture.objects.get(session_id=request.session.session_key)
        print(user_picture.edited_img.path)
        grayscale(user_picture.edited_img.path)
        return FileResponse(open(user_picture.edited_img.path, 'rb'))

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
            head, tail = os.path.split(path)
            edited_img_directory = head + "/edited_images/"
            shutil.copy(path, edited_img_directory)
            user_picture.edited_img = edited_img_directory + tail
            user_picture.save()
            #grayscale(model_instance.main_img)
            return FileResponse(open(user_picture.main_img.path, 'rb'))

def reset(request):
    if request.method == 'GET':
        user_picture = Picture.objects.get(session_id=request.session.session_key)
        return FileResponse(open(user_picture.main_img.path, 'rb'))

def index(request): 
    request.session.cycle_key()
    if not request.session.session_key:
        request.session.create()
    if not Picture.objects.filter(session_id = request.session.session_key).exists():
        user_picture = Picture.objects.create(session_id=request.session.session_key)
        path = user_picture.main_img.path
        head, tail = os.path.split(path)
        edited_img_directory = head + "/edited_images/"
        shutil.copy(path, edited_img_directory)
        user_picture.edited_img = edited_img_directory + tail
        user_picture.save()

    return render(request, 'MainPage/index.html') 
  
