from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('feed.urls')),
    path('account/', include('account.urls')),
    path('activity/', include('activity.urls')),
    path('instagram/', include('instagram_bot.urls')),
]

# Add media URL patterns only in debug mode
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)