from rest_framework.views import APIView
from .models import SavedProducts, WishlistedProducts, LikedProducts
from rest_framework.response import Response
from rest_framework import status
from feed.models import Product
from account.models import User
from .serializers import LikedProductsSerializer, SavedProductsSerializer, WishlistedProductsSerializer

class SavedProductsView(APIView):
    def post(self, request):
        serializer = SavedProductsSerializer(data=request.data)
        if serializer.is_valid():
            user = User.objects.get(id=serializer.validated_data['user_id'])
            product = Product.objects.get(id=serializer.validated_data['product_id'])
            saved_product = SavedProducts.objects.create(user=user, product=product)
            return Response({'message': 'Product saved successfully!'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def get(self, request, user_id):
        saved_products = (SavedProducts.objects
                          .filter(user_id=user_id)
                          .values_list('product_id', flat=True)
                          .distinct() 
                          .order_by('-saved_at')) 
        
        # Return just the product IDs as a list
        return Response(list(saved_products), status=status.HTTP_200_OK)

class LikedProductsView(APIView):
    def post(self, request):
        serializer = LikedProductsSerializer(data=request.data)
        if serializer.is_valid():
            user = User.objects.get(id=serializer.validated_data['user_id'])
            product = Product.objects.get(id=serializer.validated_data['product_id'])
            liked_product = LikedProducts.objects.create(user=user, product=product)
            return Response({'message': 'Product liked successfully!'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def get(self, request, user_id):
        liked_products = (LikedProducts.objects
                          .filter(user_id=user_id)
                          .values_list('product_id', flat=True)
                          .distinct() 
                          .order_by('-liked_at')) 
        return Response(list(liked_products), status=status.HTTP_200_OK)

class WishListedProductsView(APIView):
    def post(self, request):
        serializer = WishlistedProductsSerializer(data=request.data)
        if serializer.is_valid():
            user = User.objects.get(id=serializer.validated_data['user_id'])
            product = Product.objects.get(id=serializer.validated_data['product_id'])
            wishlisted_product = WishlistedProducts.objects.create(user=user, product=product)
            return Response({'message': 'Product wishlisted successfully!'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def get(self, request, user_id):
        wishlisted_products = (WishlistedProducts.objects
                          .filter(user_id=user_id)
                          .values_list('product_id', flat=True)
                          .distinct() 
                          .order_by('-wishlisted_at'))
        return Response(list(wishlisted_products), status=status.HTTP_200_OK)