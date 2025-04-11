from rest_framework import serializers
from .models import Product, Media, Post, TaggedProduct, Curation, MyntraProducts


## TODO: This might increase the latency of the similar posts API while fetching the products in the post. 
## Can be solved by creating a new API which can be triggered from PDP. 
class ProductSerializer(serializers.ModelSerializer):
    post_id = serializers.SerializerMethodField()  # Method field for the related Post ID
    class Meta:
        model = Product
        fields = [
            'id', 'category', 'subcategory', 'brand', 'name', 'image_url', 'secondary_images', 'product_link', 'price', 'discount_price', 
            'description', 'gender', 'post_id', 'created_at'
        ]
        read_only_fields = ['created_at', 'post_id'] 

    def get_post_id(self, obj):
        post = obj.posts.first()
        return post.id if post else None

    def create(self, validated_data):
        return super().create(validated_data)

class MediaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Media
        fields = "__all__"

class TaggedProductSerializer(serializers.ModelSerializer):
    product = ProductSerializer()  # Fetch and serialize the linked Product

    class Meta:
        model = TaggedProduct
        fields = ['id', 'product']


class PostSerializer(serializers.ModelSerializer):
    product = ProductSerializer(required=False)
    media = MediaSerializer(many=True, read_only=True)
    tagged_products = TaggedProductSerializer(many=True, read_only=True)  # Include tagged products

    class Meta:
        model = Post
        fields = ['id', 'post_type', 'product', 'media', 'tagged_products', 'created_at', 'title', 'description']
    
class MyntraProductsSerializer(serializers.ModelSerializer):
    class Meta:
        model = MyntraProducts
        fields = "__all__"

class CurationCreateSerializer(serializers.ModelSerializer):
    myntra_product_ids = serializers.ListField(
        child=serializers.IntegerField(),
        write_only=True,
        required=True
    )

    class Meta:
        model = Curation
        fields = ['title', 'curation_image', 'myntra_product_ids']

    def create(self, validated_data):
        myntra_product_ids = validated_data.pop('myntra_product_ids')
        curation = Curation.objects.create(**validated_data)
        
        # Add products to the curation
        products = MyntraProducts.objects.filter(id__in=myntra_product_ids)
        curation.products.add(*products)
        
        return curation

class CurationSerializer(serializers.ModelSerializer):
    products = MyntraProductsSerializer(many=True, read_only=True)
    related_curations = serializers.SerializerMethodField()

    class Meta:
        model = Curation
        fields = ['id', 'title', 'curation_image', 'products', 'related_curations']

    def get_related_curations(self, obj):
        # Get directly related curations through the ManyToMany relationship
        related_curations = obj.related_curations.all()
        return CurationSerializer(related_curations, many=True).data