from rest_framework import serializers
from .models import Product, Media, Post, TaggedProduct, Component, Curation, Item, ComponentItem


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

    

class ItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Item
        fields = "__all__"

class ComponentSerializer(serializers.ModelSerializer):
    items = serializers.SerializerMethodField()

    class Meta:
        model = Component
        fields = ['id', 'name', 'curation', 'items']

    def get_items(self, obj):
        """Retrieve all items linked to a component via ComponentItem"""
        return ItemSerializer(Item.objects.filter(item_components__component=obj), many=True).data


class CurationSerializer(serializers.ModelSerializer):
    components = ComponentSerializer(many=True, read_only=True)
    sub_curations = serializers.SerializerMethodField()

    class Meta:
        model = Curation
        fields = ['id', 'curation_type', 'curation_image', 'components', 'sub_curations']

    def get_sub_curations(self, obj):
        """Return nested curations if it is a MULTI or MULTI_INSPIRATION type"""
        if obj.curation_type in ['MULTI', 'MULTI_INSPIRATION']:
            return CurationSerializer(obj.sub_curations.all(), many=True).data
        return []