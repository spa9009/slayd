from rest_framework import serializers
from .models import SavedProducts, LikedProducts, WishlistedProducts

class SavedProductsSerializer(serializers.Serializer):
    user_id = serializers.IntegerField(write_only=True)
    product_id = serializers.IntegerField(write_only=True)

    class Meta:
        model = SavedProducts
        fields = ['user_id', 'product_id', 'saved_at']


class LikedProductsSerializer(serializers.Serializer):
    user_id = serializers.IntegerField(write_only=True)
    product_id = serializers.IntegerField(write_only=True)

    class Meta:
        model = LikedProducts
        fields = ['user_id', 'product_id', 'liked_at']


class WishlistedProductsSerializer(serializers.Serializer):
    user_id = serializers.IntegerField(write_only=True)
    product_id = serializers.IntegerField(write_only=True)

    class Meta:
        model = WishlistedProducts
        fields = ['user_id', 'product_id', 'wishlisted_at']


