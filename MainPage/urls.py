from django.urls import path

from . import views

urlpatterns = [
        path('', views.index, name='index'),
        path('upload', views.upload, name = 'upload'),
        path('call_alg', views.call_alg, name='call_alg'),
        path('download_image', views.download, name='download_image'),
]

