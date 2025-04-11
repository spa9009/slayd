from django.urls import path
from .views import PostListCreateView, PostDetailView, ProductListCreateView, ProductDetailView, ProductSearchView, RankedPostsAPIView, SimilarPostsView, CurationDetailView, MyntraProductsListCreateView, SimilarProductsView, test_aws_config, DetectedObjectProductsView, CurationCreateView

urlpatterns = [
    path('posts/', PostListCreateView.as_view(), name='post-list-create'),
    path('ranked_posts/', RankedPostsAPIView.as_view(), name='get_ranked_posts'),
    path('posts/<int:pk>/', PostDetailView.as_view(), name='post-detail'),
    path('products/', ProductListCreateView.as_view(), name='product-list-create'),
    path('products/<int:pk>/', ProductDetailView.as_view(), name='product-detail'),
    path('products/search-product-by-name/', ProductSearchView.as_view(), name='product-search-by-name'),
    path('posts/<int:post_id>/similar-posts/', SimilarPostsView.as_view(), name='similar-post'),
    path('curations/<int:pk>/', CurationDetailView.as_view(), name='curation-detail'),
    path('curations/create/', CurationCreateView.as_view(), name='curation-create'),
    path('myntra/products/', MyntraProductsListCreateView.as_view(), name='myntra-products-list-create'),
    path('similar-products/', SimilarProductsView.as_view(), name='similar-products'),
    path('detected-objects/', DetectedObjectProductsView.as_view(), name='detected-objects'),
    path('test-aws/', test_aws_config, name='test-aws-config'),
]