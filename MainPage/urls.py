from django.urls import path

from . import views

urlpatterns = [
        path('', views.index, name='index'),
        path('upload', views.upload, name = 'upload'),
        path('reset', views.reset, name = 'reset'),
        path('grayscale_button', views.grayscale_button, name = 'grayscale_button'),
        path('gaussian_noise_button', views.gaussian_noise_button, name = 'gaussian_noise_button'),
        path('salt_pepper_noise_button', views.salt_pepper_noise_button, name = 'salt_pepper_noise_button'),
]

