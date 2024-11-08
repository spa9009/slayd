from rest_framework import serializers
from .models import Product, Media, Post, TaggedProduct

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['id', 'category', 'subcategory', 'brand', 'product_name', 'product_image_url', 'product_secondary_images', 'product_link', 'price', 'discount_price', 'product_description']
        read_only_fields = ['created_at']

    def create(self, validated_data):
        # Creating the product instance with only validated data fields
        return Product.objects.create(**validated_data)

class MediaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Media
        fields = "__all__"

class TaggedProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = TaggedProduct
        fields = "__all__"

class PostSerializer(serializers.ModelSerializer):
    product = ProductSerializer(required=False)
    media = MediaSerializer(many=True, read_only=True)
    tagged_products = ProductSerializer(many=True, read_only=True)  # Include tagged products

    class Meta:
        model = Post
        fields = ['id', 'post_type', 'product', 'media', 'tagged_products', 'created_at', 'title', 'description']