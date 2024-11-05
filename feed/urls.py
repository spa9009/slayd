from django.urls import path
from .views import PostListCreateView, PostDetailView, ProductListView, ProductDetailView, ProductSearchView, RankedPostsAPIView

urlpatterns = [
    path('posts/', PostListCreateView.as_view(), name='post-list-create'),
    path('ranked_posts/', RankedPostsAPIView.as_view(), name='get_ranked_posts'),
    path('posts/<int:pk>/', PostDetailView.as_view(), name='post-detail'),
    path('products/', ProductListView.as_view(), name='product-list-create'),
    path('products/<int:pk>/', ProductDetailView.as_view(), name='product-detail'),
    path('products/search-product-by-name/', ProductSearchView.as_view(), name='product-search-by-name')
]