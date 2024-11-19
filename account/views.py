from django.contrib.auth.hashers import make_password, check_password
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import User, Brand


class SignUpView(APIView):
    def post(self, request):
        username = request.data.get('username')
        phone = request.data.get('phone')
        password = request.data.get('password')
        age = request.data.get('age')
        gender = request.data.get('gender')

        if User.objects.filter(phone=phone).exists():
            return Response({'error': 'Phone number already registered'}, status=status.HTTP_400_BAD_REQUEST)
        
        if User.objects.filter(username=username).exists():
            return Response({'error': 'Username is taken'}, status=status.HTTP_400_BAD_REQUEST)
        
        hashed_password = make_password(password)
        print(hashed_password)
        user = User.objects.create(phone=phone, password=hashed_password, username=username, age=age, gender=gender)

        return Response({
            'message': 'User registered successfully',
            'user_id' : {user.id}
            }, status=status.HTTP_201_CREATED)
    


class SignInView(APIView):
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')

        try:
            user = User.objects.get(username=username)
            print(user.password)
            if(check_password(password, user.password)):
                return Response({
                    'message': 'Login successful',
                    'user_id': {user.id}
                }, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'Invalid credentials'}, status=status.HTTP_400_BAD_REQUEST)
        except User.DoesNotExist:
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_400_BAD_REQUEST)
        
class BrandView(APIView):
    def post(self, request):
        brand_name = request.data.get('brand')

        brand = Brand.object.create(brand = brand_name)
        return Response({
            'message': 'Brand created',
            'brandId': {brand.id}
        }, status=status.HTTP_201_CREATED)