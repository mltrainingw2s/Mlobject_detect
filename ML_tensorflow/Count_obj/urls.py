from django.urls import path
from Count_obj import views
urlpatterns = [
    # path('',views.clss,name = 'counts'),
    # path('count',views.clss,name = 'obj_count'),
    path('',views.count_obj,name = 'object_count'),
   
]