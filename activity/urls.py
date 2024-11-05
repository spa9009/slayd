from django.urls import path
from .views import UserActivityView

urlpatterns = [
    path('activity/', UserActivityView.as_view(), name='user-activity'),
]