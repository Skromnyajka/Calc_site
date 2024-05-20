from django.urls import path

from . import views

urlpatterns = [
    path("", views.index),
    path("inter", views.int_index),
    path("code", views.code),
    path("done", views.done),
]


