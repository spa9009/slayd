from rest_framework import generics, status
from .models import Product, Post, TaggedProduct, Media
from .serializers import ProductSerializer, PostSerializer
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.db.models import Q
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
        tagged_product_ids = self.request.data.get('tagged_products', [])
        media = self.request.data.get('media', [])
        
        if post_type == 'PRODUCT_POST' and product_id:
            product = Product.objects.get(id=product_id)
            serializer.save(product=product, post_type='PRODUCT_POST')
        elif post_type == 'TAGGED_POST':
            post = serializer.save(post_type='TAGGED_POST')
            if tagged_product_ids:
                tagged_products = Product.objects.filter(id__in=tagged_product_ids)
                for product in tagged_products:
                    TaggedProduct.objects.create(post=post, product=product)
            if media:
                for media_item in media:
                    Media.objects.create(post=post, type=media_item['type'], url=media_item['url'])
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

        # Fetch all posts once
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
    

class SimilarPostsView(APIView):
    def get(self, request, post_id) :
        try:
            target_post = Post.objects.get(id=post_id)

            post_brand = None
            categories = []
            sub_categories = []

            tagged_products = target_post.tagged_products.all()
            if len(tagged_products.all()) > 0: 
                post_brand = tagged_products.first().product.brand
                print(post_brand)
                for tagged_product in tagged_products: 
                    print(tagged_product.product.category)
                    categories.append(tagged_product.product.category)
                    sub_categories.append(tagged_product.product.subcategory)
            else :
                post_brand = target_post.product.brand
                categories.append(target_post.product.category)
                sub_categories.append(target_post.product.subcategory)

            similar_brand_posts = Post.objects.filter(
                Q(product__brand=post_brand) | 
                Q(tagged_products__product__brand=post_brand)
            ).exclude(id=target_post.id).distinct()

            similar_subcategory_posts = Post.objects.filter(
                Q(product__subcategory__in=sub_categories) |
                Q(tagged_products__product__subcategory__in=sub_categories)
            ).exclude(id__in=similar_brand_posts).exclude(id=target_post.id).distinct()

            similar_category_posts = Post.objects.filter(
                Q(product__category__in=categories) |
                Q(tagged_products__product__category__in=categories)
            ).exclude(id__in=similar_brand_posts).exclude(id__in=similar_subcategory_posts).exclude(id=target_post.id).distinct()

            combined_post = []

            all_similar_posts_product = list(similar_brand_posts.filter(post_type='PRODUCT_POST')) + list(similar_subcategory_posts.filter(post_type='PRODUCT_POST')) + list(similar_category_posts.filter(post_type='PRODUCT_POST'))

            all_similar_posts_tagged = list(similar_brand_posts.filter(post_type='TAGGED_POST')) + \
                list(similar_subcategory_posts.filter(post_type='TAGGED_POST')) + \
                list(similar_category_posts.filter(post_type='TAGGED_POST'))
            print(f"Size of tagged_product list: {len(all_similar_posts_tagged)}")


            while all_similar_posts_tagged or all_similar_posts_product: 
                temp_list = []
                temp_list.extend(all_similar_posts_product[:5])
                all_similar_posts_product = all_similar_posts_product[5:]

                temp_list.extend(all_similar_posts_tagged[:5])
                all_similar_posts_tagged = all_similar_posts_tagged[5:]
                random.shuffle(temp_list)
                
                combined_post.extend(temp_list)


            print(f"Size of combined_post list: {len(combined_post)}")

            page_number = request.query_params.get('page', 1)
            paginator = Paginator(combined_post, 10)

            try:
                paginated_posts = paginator.page(page_number)
            except:
                return Response({"error": "Invalid page number"}, status=400)

            serializer = PostSerializer(paginated_posts, many=True)

            return Response({
                "current_page": paginated_posts.number,
                "total_pages": paginator.num_pages,
                "results": serializer.data,
            })
        
        except Post.DoesNotExist:
            return Response({
                "error": "Post not found"
            }, status=status.HTTP_404_NOT_FOUND)
        # except Exception as e:
        #     return Response({
        #         "error": "Error while fetching similar posts",
        #     }, status=status.HTTP_404_NOT_FOUND)

        

