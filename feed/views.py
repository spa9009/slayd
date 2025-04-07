from rest_framework import generics, status
from .models import Product, Post, TaggedProduct, Media, Curation, MyntraProducts, DetectedObjectProducts
from .serializers import ProductSerializer, PostSerializer, CurationSerializer, MyntraProductsSerializer
from rest_framework.response import Response
from rest_framework.views import APIView
from django.core.paginator import Paginator
from django.db.models import Q
import random
import logging
from .utils.similarity_search import SimilaritySearcher
import requests
from utils.metrics import MetricsUtil
import boto3
from rest_framework.decorators import api_view

logger = logging.getLogger(__name__)

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
        # Get the search query from the request parameters
        product_name = request.query_params.get('name', '').strip()

        if not product_name:
            return Response(
                {'error': 'The "name" query parameter is required.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Retrieve the first product matching the query
        product = Product.objects.filter(name__icontains=product_name).first()

        if product:
            serializer = self.get_serializer(product)
            return Response(serializer.data, status=status.HTTP_200_OK)

        return Response({'message': 'No product found'}, status=status.HTTP_404_NOT_FOUND)
        
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
            added_post_ids = set()

            all_similar_posts_product = list(similar_subcategory_posts.filter(post_type='PRODUCT_POST')) + \
                list(similar_category_posts.filter(post_type='PRODUCT_POST')) + \
                list(similar_brand_posts.filter(post_type='PRODUCT_POST'))

            all_similar_posts_tagged = list(similar_subcategory_posts.filter(post_type='TAGGED_POST')) + \
                list(similar_category_posts.filter(post_type='TAGGED_POST')) + \
                list(similar_brand_posts.filter(post_type='TAGGED_POST'))
            print(f"Size of tagged_product list: {len(all_similar_posts_tagged)}")


            while all_similar_posts_tagged or all_similar_posts_product:
                temp_list = []

                for post in all_similar_posts_product[:7]:
                    if post.id not in added_post_ids:
                        temp_list.append(post)
                        added_post_ids.add(post.id)
                all_similar_posts_product = all_similar_posts_product[7:]

                for post in all_similar_posts_tagged[:3]:
                    if post.id not in added_post_ids:
                        temp_list.append(post)
                        added_post_ids.add(post.id)
                all_similar_posts_tagged = all_similar_posts_tagged[3:]

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

class CurationDetailView(generics.RetrieveAPIView):
    queryset = Curation.objects.prefetch_related(
        'components__component_items__item',  # Prefetch items within components
        'sub_curations__components__component_items__item'  # Prefetch items for sub_curations
    )
    serializer_class = CurationSerializer

class MyntraProductsListCreateView(generics.ListCreateAPIView):
    queryset = MyntraProducts.objects.all()
    serializer_class = MyntraProductsSerializer

class SimilarProductsView(APIView):
    @MetricsUtil.track_execution_time('SimilarProductsView')
    def get(self, request):
        logger = logging.getLogger(__name__)
        try:
            image_url = request.query_params.get('image_url')
            search_type = request.query_params.get('search_type', 'combined_75')
            page = int(request.query_params.get('page', 1))
            items_per_page = int(request.query_params.get('items_per_page', 20))
            
            # Get crop parameters if provided
            x = request.query_params.get('x')
            y = request.query_params.get('y')
            width = request.query_params.get('width')
            height = request.query_params.get('height')
            
            # Log if crop parameters are provided
            if all(param is not None for param in [x, y, width, height]):
                logger.info(f"Crop parameters provided: x={x}, y={y}, width={width}, height={height}")
            
            if not image_url:
                MetricsUtil.record_failure('SimilarProductsView', 'MissingImageURL')
                return Response(
                    {"error": "image_url parameter is required"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get similar products
            try:
                searcher = SimilaritySearcher()
                similar_products = searcher.get_similar_products(
                    image_url, 
                    search_type=search_type,
                    page=page,
                    items_per_page=items_per_page,
                    top_k=120,
                    x=x,
                    y=y,
                    width=width,
                    height=height
                )
                
                MetricsUtil.record_success('SimilarProductsView', [
                    {'Name': 'SearchType', 'Value': search_type},
                    {'Name': 'Page', 'Value': str(page)}
                ])
                return Response(similar_products)
            except RuntimeError as e:
                logger.error(f"Runtime error in similarity search: {str(e)}")
                return Response(
                    {"error": str(e)}, 
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY
                )
            except Exception as e:
                MetricsUtil.record_failure('SimilarProductsView', type(e).__name__)
                raise

        except ValueError as e:
            MetricsUtil.record_failure('SimilarProductsView', 'InvalidPaginationParameters')
            return Response(
                {"error": f"Invalid pagination parameters: {str(e)}"}, 
                status=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE
            )
        except Exception as e:
            MetricsUtil.record_failure('SimilarProductsView', 'UnhandledException')
            logger.error(f"Unhandled exception: {str(e)}")
            return Response(
                {"error": "Internal server error"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@api_view(['GET'])
def test_aws_config(request):
    logger = logging.getLogger(__name__)
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        
        # Test if we can list CloudWatch metrics
        cw = session.client('cloudwatch')
        metrics = cw.list_metrics(Namespace='AWS/EC2')
        
        return Response({
            'status': 'success',
            'aws_config': {
                'region': session.region_name,
                'has_credentials': credentials is not None,
                'credential_type': type(credentials).__name__ if credentials else None,
                'can_list_metrics': bool(metrics),
            }
        })
    except Exception as e:
        logger.error("AWS configuration test failed", exc_info=True)
        return Response({
            'status': 'error',
            'error': str(e)
        }, status=500)

class DetectedObjectProductsView(APIView):
    """
    API endpoint to retrieve detected objects and their similar products for an image.
    
    GET parameters:
    - image_url (required): URL of the image to get detected objects for
    - object_id (optional): ID of a specific object to filter results
    - page (optional): Page number for pagination (default: 1)
    - items_per_page (optional): Number of products per page (default: 20)
    """
    
    def get(self, request):
        logger.info("Fetching detected objects and similar products")
        
        try:
            # Get query parameters
            image_url = request.query_params.get('image_url')
            object_id = request.query_params.get('object_id')
            page = int(request.query_params.get('page', 1))
            items_per_page = int(request.query_params.get('items_per_page', 20))
            
            if not image_url:
                return Response(
                    {"error": "image_url parameter is required"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get all detected objects for this image
            objects = DetectedObjectProducts.objects.filter(
                image_url=image_url,
                is_whole_image=False
            )
            
            # Get whole image record if it exists
            whole_image_record = DetectedObjectProducts.objects.filter(
                image_url=image_url,
                is_whole_image=True
            ).first()
            
            # If no objects or whole image record found, return error
            if not objects.exists() and not whole_image_record:
                return Response(
                    {"error": "No results found for this image"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # If object_id is provided, filter for that specific object
            if object_id:
                objects = objects.filter(id=object_id)
                
                if not objects.exists():
                    return Response(
                        {"error": f"Object with ID {object_id} not found for this image"}, 
                        status=status.HTTP_404_NOT_FOUND
                    )
                
                # Get the object
                obj = objects.first()
                all_product_ids = obj.similar_products
                total_products = len(all_product_ids)
                total_pages = (total_products + items_per_page - 1) // items_per_page if total_products > 0 else 1
                
                # Calculate start and end indices
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_products)
                
                # Get the products for this page
                paginated_products = all_product_ids[start_idx:end_idx]
                
                # Get product details
                product_details = []
                for product_id in paginated_products:
                    try:
                        product = MyntraProducts.objects.get(id=product_id)
                        product_details.append({
                            'id': product.id,
                            'name': product.name,
                            'brand': product.brand,
                            'price': product.price,
                            'discount_price': product.discount_price,
                            'image_url': product.image_url,
                            'product_link': product.product_link,
                            'marketplace': product.marketplace
                        })
                    except MyntraProducts.DoesNotExist:
                        logger.warning(f"Product with ID {product_id} not found")
                
                # Format response with pagination
                response_data = {
                    'object': {
                        'id': obj.id,
                        'label': obj.label,
                        'x': obj.x,
                        'y': obj.y,
                        'width': obj.width,
                        'height': obj.height,
                        'confidence': obj.confidence,
                        'products': product_details,
                        'pagination': {
                            'current_page': page,
                            'total_pages': total_pages,
                            'total_items': total_products,
                            'items_per_page': items_per_page
                        }
                    }
                }
                
                return Response(response_data)
            
            else:
                # No specific object requested, return all objects' info
                
                # Initialize response data structure
                response_data = {
                    'objects': []
                }
                
                # Add detected objects info
                for obj in objects:
                    response_data['objects'].append({
                        'id': obj.id,
                        'label': obj.label,
                        'x': obj.x,
                        'y': obj.y,
                        'width': obj.width,
                        'height': obj.height,
                        'confidence': obj.confidence,
                        'products_count': len(obj.similar_products)
                    })
                
                # Only include whole image products if no objects were detected
                if not objects.exists() and whole_image_record:
                    # Get products for the whole image
                    whole_image_products = whole_image_record.similar_products
                    
                    # For whole image products, apply pagination
                    total_products = len(whole_image_products)
                    total_pages = (total_products + items_per_page - 1) // items_per_page if total_products > 0 else 1
                    
                    # Calculate start and end indices
                    start_idx = (page - 1) * items_per_page
                    end_idx = min(start_idx + items_per_page, total_products)
                    
                    # Get the products for this page
                    paginated_products = whole_image_products[start_idx:end_idx]
                    
                    # Get product details
                    product_details = []
                    for product_id in paginated_products:
                        try:
                            product = MyntraProducts.objects.get(id=product_id)
                            product_details.append({
                                'id': product.id,
                                'name': product.name,
                                'brand': product.brand,
                                'price': product.price,
                                'discount_price': product.discount_price,
                                'image_url': product.image_url,
                                'product_link': product.product_link,
                                'marketplace': product.marketplace
                            })
                        except MyntraProducts.DoesNotExist:
                            logger.warning(f"Product with ID {product_id} not found")
                    
                    # Add whole image data to response
                    response_data['whole_image'] = {
                        'products': product_details,
                        'pagination': {
                            'current_page': page,
                            'total_pages': total_pages,
                            'total_items': total_products,
                            'items_per_page': items_per_page
                        }
                    }
                    
                return Response(response_data)
                
        except ValueError as e:
            return Response(
                {"error": f"Invalid pagination parameters: {str(e)}"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.exception("Error fetching detected objects")
            return Response(
                {"error": f"Internal server error: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
