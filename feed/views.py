from rest_framework import generics, status
from .models import Product, Post, TaggedProduct
from .serializers import ProductSerializer, PostSerializer
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import JsonResponse
import random

class ProductListCreateView(generics.ListCreateAPIView):
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

class ProductSearchView(generics.GenericAPIView):
    serializer_class = ProductSerializer

    def get(self, request, *args, **kwargs):
        try:
            # Get the search query from the request parameters
            product_name = request.query_params.get('name', None)

            if product_name:
                # Filter products by name (case-insensitive)
                products = Product.objects.filter(name__icontains=product_name)

                if products.exists():
                    # Serialize the products and return the response
                    serializer = self.get_serializer(products, many=True)
                    return Response(serializer.data, status=status.HTTP_200_OK)
                else:
                    return Response({'message': 'No products found'}, status=status.HTTP_404_NOT_FOUND)
            else:
                return Response({'error': 'product_name parameter is required'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print('Exception is', e)
        
class RankedPostsAPIView(APIView):
    def get(self, request):
        request_size = int(request.query_params.get('request_size', 10))

        all_posts = list(Post.objects.all())

        tagged_posts = [post for post in all_posts if post.post_type == 'TAGGED_POST']
        normal_posts = [post for post in all_posts if post.post_type != 'TAGGED_POST']

        random.shuffle(tagged_posts)
        random.shuffle(normal_posts)

        tagged_count = request_size // 2
        normal_count = request_size - tagged_count

        selected_tagged_posts = tagged_posts[:tagged_count]
        selected_normal_posts = normal_posts[:normal_count]

        ranked_posts = selected_tagged_posts + selected_normal_posts

        random.shuffle(ranked_posts)

        serializer = PostSerializer(ranked_posts, many=True)
        return Response(serializer.data)