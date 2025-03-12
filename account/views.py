from django.contrib.auth.hashers import make_password, check_password
from rest_framework.views import APIView, View
from rest_framework.response import Response
from rest_framework import status
from .models import UserRecord, Brand, UserPreferences
from .serializers import UserPreferencesSerializer
from django.shortcuts import get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
from utils.metrics import MetricsUtil

class SignUpView(APIView):
    def post(self, request):
        username = request.data.get('username')
        phone = request.data.get('phone')
        password = request.data.get('password')
        age = request.data.get('age')
        gender = request.data.get('gender')

        if UserRecord.objects.filter(phone=phone).exists():
            return Response({'error': 'Phone number already registered'}, status=status.HTTP_400_BAD_REQUEST)
        
        if UserRecord.objects.filter(username=username).exists():
            return Response({'error': 'Username is taken'}, status=status.HTTP_400_BAD_REQUEST)
        
        hashed_password = make_password(password)
        print(hashed_password)
        user = UserRecord.objects.create(phone=phone, password=hashed_password, username=username, age=age, gender=gender)

        return Response({
            'message': 'User registered successfully',
            'user_id' : {user.id}
            }, status=status.HTTP_201_CREATED)
    


class SignInView(APIView):
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')

        try:
            user = UserRecord.objects.get(username=username)
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
    

class UserPreferenceView(APIView) :
    serializer_class = UserPreferencesSerializer

    def get_serializer(self, *args, **kwargs):
        return self.serializer_class(*args, **kwargs)

    def get(self, request, *args, **kwargs):
        user_id = request.query_params.get("user_id")

        if not user_id:
            return Response({"detail": "User ID is required."}, status=status.HTTP_400_BAD_REQUEST)

        profile = get_object_or_404(UserPreferences, user_id=user_id)
        serializer = self.get_serializer(profile)
        return Response(serializer.data)
    
    def post(self, request, *args, **kwargs):
        user_id = request.data.get("user_id")
        avoid_styles = request.data.get("avoid_styles", [])
        liked_aesthetics = request.data.get("aesthetics", [])

        if not user_id:
            return Response({"detail": "User ID is required."}, status=status.HTTP_400_BAD_REQUEST)

        preference, created = UserPreferences.objects.get_or_create(user_id=user_id)

        # Update fields
        preference.avoid_styles = avoid_styles
        preference.aesthetics = liked_aesthetics  # JSONField
        preference.save()

        serializer = self.get_serializer(preference)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    

class MetaWebhookView(View):
    @MetricsUtil.track_execution_time('MetaWebhookVerification')
    def get(self, request):
        VERIFY_TOKEN = "slayd"
        mode = request.GET.get('hub.mode')
        token = request.GET.get('hub.verify_token')
        challenge = request.GET.get('hub.challenge')

        if mode and token:
            if mode == 'subscribe' and token == VERIFY_TOKEN:
                MetricsUtil.record_success('MetaWebhookVerification', [
                    {'Name': 'VerificationType', 'Value': 'Challenge'}
                ])
                return HttpResponse(challenge)
            else:
                MetricsUtil.record_failure('MetaWebhookVerification', 'InvalidToken', [
                    {'Name': 'Mode', 'Value': mode}
                ])
                return HttpResponse('Forbidden', status=403)

        MetricsUtil.record_failure('MetaWebhookVerification', 'MissingParameters')
        return HttpResponse('Bad Request', status=400)

    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    @MetricsUtil.track_execution_time('MetaWebhookEvent')
    def post(self, request):
        try:
            data = json.loads(request.body)
            
            # Extract event type if available in the webhook payload
            event_type = data.get('entry', [{}])[0].get('changes', [{}])[0].get('field', 'unknown')
            
            MetricsUtil.put_metric(
                'WebhookEventReceived',
                1,
                [
                    {'Name': 'Endpoint', 'Value': 'MetaWebhook'},
                    {'Name': 'EventType', 'Value': event_type}
                ]
            )
            
            print("Webhook received:", data)
            MetricsUtil.record_success('MetaWebhookEvent', [
                {'Name': 'EventType', 'Value': event_type}
            ])
            return JsonResponse({'status': 'success'})
            
        except json.JSONDecodeError:
            MetricsUtil.record_failure('MetaWebhookEvent', 'InvalidJSON')
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            MetricsUtil.record_failure('MetaWebhookEvent', type(e).__name__)
            return JsonResponse({'error': 'Internal Server Error'}, status=500)