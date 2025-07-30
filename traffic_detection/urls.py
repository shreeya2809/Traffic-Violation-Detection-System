# traffic_detection/urls.py

from django.contrib import admin
from django.urls import path, include  # ✅ include imported
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('app1.urls')),  # ✅ include used here only
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
