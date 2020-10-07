from django.urls import path
from . import views

urlpatterns = [
    # /blog/
    path('', views.index, name='index'),
    path('post/<int:pk>', views.PostDetailView.as_view(), name='post'),
    path('post/create', views.PostCreate.as_view(), name='post_create'),
    # path('post/record', views.button, name='record'),
]
