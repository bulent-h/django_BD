from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_images, name='upload_images'),
    path('list_results', views.list_results, name='list_results'),
    path('result/<int:pk>/', views.show_result, name='result'),
    path('edit/<int:pk>/', views.edit_result, name='edit'),
    path('delete/<int:pk>/', views.delete_result, name='delete'),
    ]


