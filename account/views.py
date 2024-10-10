from django.contrib.auth.hashers import make_password, check_password
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import User


class SignUpView(APIView):
    def post(self, request):
        email = request.data.get('email')
        username = request.data.get('username')
        phone = request.data.get('phone')
        password = request.data.get('password')

        if User.objects.filter(phone=phone).exists():
            return Response({'error': 'Phone number already registered'}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(email=email).exists():
            return Response({'error': 'Email already registered'}, status=status.HTTP_400_BAD_REQUEST)
        
        if User.objects.filter(username=username).exists():
            return Response({'error': 'Username is taken'}, status=status.HTTP_400_BAD_REQUEST)
        
        hashed_password = make_password(password)
        print(hashed_password)
        user = User.objects.create(email=email, phone=phone, password=hashed_password, username=username)

        return Response({'message': 'User registered successfully'}, status=status.HTTP_201_CREATED)
    


class SignInView(APIView):
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')

        try:
            user = User.objects.get(username=username)
            print(user.password)
            if(check_password(password, user.password)):
                return Response({'message': 'Login successful'}, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'Invalid credentials'}, status=status.HTTP_400_BAD_REQUEST)
        except User.DoesNotExist:
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_400_BAD_REQUEST)