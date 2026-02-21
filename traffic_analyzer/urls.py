"""
URL configuration for traffic_analyzer project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from django.http import FileResponse, Http404
import os

def media_access(request, path):
    """
    Custom view to serve MEDIA files dynamically in production
    since WhiteNoise only supports static files easily.
    Injects CORS headers so the Vercel React app can play the video.
    """
    file_path = os.path.join(settings.MEDIA_ROOT, path)
    if not os.path.exists(file_path):
        raise Http404("Media file not found")
        
    response = FileResponse(open(file_path, 'rb'))
    response['Access-Control-Allow-Origin'] = '*'
    return response

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('analysis.urls')),
    path('media/<path:path>', media_access),
]
