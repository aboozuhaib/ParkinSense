from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name = "index"),
    path('datasets/', views.dataset, name="dataset"),
    path('datainput/', views.upload_csv, name='upload_csv'),
]