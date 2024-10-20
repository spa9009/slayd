from rest_framework import generics
from .models import Product, Post, TaggedProduct
from .serializers import ProductSerializer, PostSerializer

class ProductListView(generics.ListCreateAPIView):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

# View to retrieve, update, and delete a specific product
class ProductDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

class PostDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer

class PostListCreateView(generics.ListCreateAPIView):
    queryset = Post.objects.all().order_by('-created_at')
    serializer_class = PostSerializer

    def perform_create(self, serializer):
        post_type = self.request.data.get('post_type')
        product_id = self.request.data.get('product')
        tagged_product_ids = self.request.data.get('tagged_products')
        
        if post_type == 'PRODUCT_POST' and product_id:
            product = Product.objects.get(id=product_id)
            serializer.save(product=product, post_type='PRODUCT_POST')
        elif post_type == 'SOCIAL_POST':
            post = serializer.save(post_type='SOCIAL_POST')
            if tagged_product_ids:
                for product_id in tagged_product_ids:
                    product = Product.objects.get(id=product_id)
                    TaggedProduct.objects.create(post=post, product=product)
        else:
            serializer.save(post_type=post_type)