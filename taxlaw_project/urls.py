from django.urls import path

from taxlaw_project import views

app_name='taxlaw_project'
urlpatterns = [
    path('', views.first_page),
    path('classification/', views.class_func),
]