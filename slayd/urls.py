from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse

def health_check(request):
    return JsonResponse({"status": "ok"}, status=200)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('feed.urls')),
    path('account/', include('account.urls')),
    path('activity/', include('activity.urls')),
    path('instagram/', include('instagram_bot.urls')),
    path('vision/', include('vision.urls')),
    path("health/", health_check)
]