from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),

    # Image captioning routes
    path('api/', include('image_caption.urls')),

    # Sentiment analysis routes
    path('api/', include('sentiment.urls')),

    # Combined route: upload image → caption + sentiment
    path('api/', include('combined.urls')), 
]
