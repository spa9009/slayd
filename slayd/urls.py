from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('feed.urls')),
    path('account/', include('account.urls')),
    path('activity/', include('activity.urls')),
]