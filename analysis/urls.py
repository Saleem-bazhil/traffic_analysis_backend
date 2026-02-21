from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'analysis', views.AnalysisViewSet, basename='analysis')

urlpatterns = [
    path('analysis/upload/', views.AnalysisUploadView.as_view(), name='analysis-upload'),
    path('signal-config/', views.SignalConfigView.as_view(), name='signal-config'),
    path('', include(router.urls)),
]
