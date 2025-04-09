from django.urls import path
from . import views

urlpatterns = [
    path('detect/', views.detect_objects, name='detect_objects'),
] 