from django.urls import path
from .views import UserActivityView, FollowView, ImageUploadView

urlpatterns = [
    path('activity/', UserActivityView.as_view(), name='user-activity'),
    path('follow/', FollowView.as_view(), name='user-follow'),
    path('upload/', ImageUploadView.as_view(), name='image-upload')
]