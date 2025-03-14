from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.models import AnonymousUser
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from feed.models import Post
from account.models import UserRecord
from .models import UserActivity, Follow
from .serializers import UserActivitySerializer, FollowSerializer
import logging
from django.utils import timezone

logger = logging.getLogger(__name__)

class UserActivityView(generics.GenericAPIView):
    serializer_class = UserActivitySerializer

    def get_queryset(self):
        # Filtering by user_id and action if provided
        user_id = self.request.query_params.get("user_id")
        action = self.request.query_params.get("action")
        
        queryset = UserActivity.objects.filter(user_id=user_id)
        if action:
            queryset = queryset.filter(action=action)
        
        return queryset.order_by('-timestamp')
    
    def get(self, request, *args, **kwargs):
        user_id = request.query_params.get("user_id")
        
        if not user_id:
            return Response(
                {"detail": "User ID is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    def post(self, request, *args, **kwargs):
        # Create a new interaction based on user_id, product_id, and action provided in the request data
        print(request)
        print('Reached here')
        print(request.data)
        user_id = request.data.get("user_id")
        post_id = request.data.get("post_id")
        action = request.data.get("action")

        if not (user_id and post_id and action):
            return Response(
                {"detail": "User ID, Post ID, and action are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = get_object_or_404(UserRecord, id=user_id)
        post = get_object_or_404(Post, id=post_id)

        # Create interaction data using PK values
        interaction_data = {
            "user": user_id,  # Use user_id as PK
            "post": post_id,  # Use post_id as PK
            "action": action
        }

        # Validate with the serializer using the correct data
        serializer = self.get_serializer(data=interaction_data)
        serializer.is_valid(raise_exception=True)

        # Now create the UserActivity instance based on validated data
        activity = serializer.save(timestamp=timezone.now())  # Assuming you want to save the current timestamp

        # Return the created activity data
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
class FollowView(generics.CreateAPIView) :
    serializer_class = FollowSerializer
    def post(self, request):

        data = {
            "user":request.data.get("user_id"),
            "publisher_type":request.data.get("publisher_type"),
            "publisher":request.data.get("publisher")
        }

        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)

        # Now create the UserActivity instance based on validated data
        follow = serializer.save()

        return Response(serializer.data, status=status.HTTP_201_CREATED)