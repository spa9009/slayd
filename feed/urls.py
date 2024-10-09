from django.urls import path
from .views import ProductFeedView, ProductCreateView

urlpatterns = [
    path('feed/', ProductFeedView.as_view(), name='product-feed'),
    path('product/add/', ProductCreateView.as_view(), name='product-create')
]