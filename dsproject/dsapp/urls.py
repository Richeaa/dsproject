from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('underperform/', views.underperform, name='underperform_students'),
    path('predict-grades/', views.predict_status, name='predict_status'),
]