from rest_framework import serializers
from .models import Product, Media, Post, TaggedProduct

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = "__all__"

class MediaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Media
        fields = "__all__"

class TaggedProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = TaggedProduct
        fields = "__all__"

class PostSerializer(serializers.ModelSerializer):
    product = ProductSerializer(many=False, required=False)  
    media = MediaSerializer(many=True, read_only=True)
    tagged_products = ProductSerializer(many=True, read_only=True)  # Include tagged products

    class Meta:
        model = Post
        fields = ['id', 'post_type', 'product', 'media', 'tagged_products', 'created_at', 'title', 'description']