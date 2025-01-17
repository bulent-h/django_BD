
from django.contrib import admin
from django.urls import include, path

from django.conf.urls.static import static
from django.conf import settings

from core import views
from core import urls
 
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',include('core.urls'))
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
