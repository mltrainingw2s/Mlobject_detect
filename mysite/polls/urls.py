from django.contrib import admin
from django.urls import path
from flows import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.get_all,name='get_all')
]