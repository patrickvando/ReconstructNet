from django.db import models

# Create your models here.
class Picture(models.Model):
    # add migrations
    main_img = models.ImageField(upload_to='images/', default='images/default.jpg')
    edited_img = models.ImageField(blank=True, default=None)
    session_id = models.TextField(blank=True, default=None)
