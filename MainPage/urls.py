from django.urls import path

from . import views

urlpatterns = [
        path('', views.index, name='index'),
        path('upload', views.upload, name = 'upload'),
        path('reset', views.reset, name = 'reset'),
        path('grayscale_button', views.grayscale_button, name = 'grayscale_button'),

]

