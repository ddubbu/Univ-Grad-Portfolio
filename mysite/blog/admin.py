from django.contrib import admin
from .models import Category, Post

# from .models import Graph
# # Graph 클래스를 inline으로 나타냄
# class GraphInline(admin.TabularInline):
#     model = Graph
#
# # Post 클래스는 해당하는 Photo 객체를 리스트로 관리
# class PostAdmin(admin.ModelAdmin):
#     inlines = [GraphInline, ]

# Register your models here.
admin.site.register(Category)
admin.site.register(Post)
# admin.site.register(PostAdmin)