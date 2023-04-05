from django.urls import path

from pill import views


app_name = 'pill'
urlpatterns = [
    path('detector/', views.PillDetectorAPIView.as_view(), name='pill_detector_api'),
    path('detector/test/', views.pill_detector_test, name='pill_detector_test'),
]