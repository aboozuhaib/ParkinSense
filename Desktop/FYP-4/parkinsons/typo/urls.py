from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name = "index"),
    path('about/', views.about, name='about'),
    path('<str:type>/datasets/', views.dataset, name="dataset"),
    path('<str:type_>/<str:class_>/datainput/', views.upload_csv, name='upload_csv'),
]