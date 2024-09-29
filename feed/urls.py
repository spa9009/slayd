from django.urls import path
from .views import ProductFeedView

urlpatterns = [
    path('feed/', ProductFeedView.as_view(), name='product-feed'),
]