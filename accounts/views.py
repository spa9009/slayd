from django.contrib.auth.hashers import make_password, check_password
from rest_framework import status
from rest_framework.views import APIView
from models.user import User
from rest_framework.response import Response
import time

# Create your views here.
class SignUpView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        phone = request.data.get('phone')
        username = request.data.get('username')

        if User.objects.filter(phone=phone).exists():
            return Response({'error': 'Phone number already registered'} , 
                            status=status.HTTP_400_BAD_REQUEST)
        
        if User.objects.filter(email=email).exists():
            return Response({'error': 'Email already registered'} , 
                            status=status.HTTP_400_BAD_REQUEST)
        
        hashed_password = make_password(password)
        user = User.objects.create(email=email, 
                                   phone=phone, 
                                   username=username, 
                                   password=hashed_password, 
                                   signup_timestamp=int(time.time() * 1000))
        return Response({'message': 'User registered successfully'}, status=status.HTTP_201_CREATED)
    


class SignInView(APIView):
    def post(self, request):
        phone = request.data.get('phone')
        password = request.data.get('password')

        try:
            user = User.objects.get(phone=phone)

            if check_password(password, user.password):
                return Response({'message': 'Login successful'}, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'Invalid credentials'}, status=status.HTTP_400_BAD_REQUEST)
        except User.DoesNotExist:
            return Response({'error': 'Invalid credentials. User does not exist',
                             'errorCode': '401'}, 
                             status=status.HTTP_400_BAD_REQUEST)