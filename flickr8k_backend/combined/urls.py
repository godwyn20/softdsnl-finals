from django.urls import path
from . import views

urlpatterns = [
    path("predict_image_sentiment/", views.predict_image_sentiment, name="predict_image_sentiment"),
]
