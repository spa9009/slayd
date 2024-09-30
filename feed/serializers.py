from rest_framework import serializers
from .models import Product

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['product_id', 'category', 'subcategory', 'brand', 'product_name', 'product_image_url', 'product_secondary_images', 'product_link', 'price', 'created_at']