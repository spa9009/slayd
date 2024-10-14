from django.urls import path
from .views import SavedProductsView, LikedProductsView, WishListedProductsView

urlpatterns = [
    path('save/', SavedProductsView.as_view(), name='save_product'),
    path('save/<int:user_id>/', SavedProductsView.as_view(), name='get_saved_products'),
    path('like/', LikedProductsView.as_view(), name='like_product'),
    path('like/<int:user_id>/', LikedProductsView.as_view(), name='get_liked_products'),
    path('wishlist/', WishListedProductsView.as_view(), name='wishlist_product'),
    path('wishlist/<int:user_id>/', WishListedProductsView.as_view(), name='get_wishlisted_products')
]