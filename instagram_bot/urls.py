from django.urls import path
from .views import MetaWebhookView, VideoWebhookView, VideoPostListView

urlpatterns = [
    path('webhook/', MetaWebhookView.as_view(), name='instagram_webhook'),
    path('video-webhook/', VideoWebhookView.as_view(), name='video-webhook'),
    path('video-posts/', VideoPostListView.as_view(), name='video-posts'),
]
