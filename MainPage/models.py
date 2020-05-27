from django.db import models
import os

def main_img_directory(instance, filename):
    return os.path.join('images', 'user_images', instance.session_id, filename)

def edited_img_directory(instance, filename):
    return os.path.join('images', 'user_images', instance.session_id, 'edited', filename)

class Picture(models.Model):
    main_img = models.ImageField(upload_to=main_img_directory, default=None)
    edited_img = models.ImageField(upload_to=edited_img_directory, blank=True, default=None)
    session_id = models.TextField(blank=True, default=None)
