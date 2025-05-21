from django.urls import path
from . import views
from . import admin_view

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('view_learning_path/', views.view_learning_path, name='view_learning_path'),
    path('get-learning-path-data/', views.get_learning_path_data, name='get_learning_path_data'),
    path('underperform/', views.underperform, name='underperform_students'),
    path('predict-grades/', views.predict_status, name='predict_status'),
    path('retrain-model/<int:model_id>/', admin_view.retrain_model_view, name='retrain_model'),
    path('predict/', views.predict_cluster, name='predict_cluster'),
    path('search/', views.search_cluster, name='search_cluster'),
]