from django.urls import path
from . import views
from . import admin_view

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('view_learning_path/', views.view_learning_path, name='view_learning_path'),
    path('get-learning-path-data/', views.get_learning_path_data, name='get_learning_path_data'),
    path('optimal/', views.optimal, name='optimal_schedule'),
    path('predict-schedule/', views.predict_schedule, name='predict_schedule'),
    path('retrain-model/<int:model_id>/', admin_view.retrain_model_view, name='retrain_model'),
    path('predict/', views.predict_cluster, name='predict_cluster'),
    path('search/', views.search_cluster, name='search_cluster'),
]