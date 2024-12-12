from rest_framework import serializers
from .models import UserActivity, Follow
from feed.models import Post
from feed.serializers import PostSerializer

class UserActivitySerializer(serializers.ModelSerializer):
    post = serializers.PrimaryKeyRelatedField(queryset=Post.objects.all())    
    class Meta: 
        model = UserActivity
        fields = ['user', 'post', 'action', 'timestamp']
    
    def validate_action(self, value):
        if value not in dict(UserActivity.ACTIONS):
            raise serializers.ValidationError('Invalid action type')
        return value
    
    def to_representation(self, instance):
        representation = super().to_representation(instance)
        representation['post'] = PostSerializer(instance.post).data
        return representation
    
class FollowSerializer(serializers.ModelSerializer):
    class Meta: 
        model = Follow
        fields = ['user', 'publisher_type', 'publisher', 'timestamp']
        read_only_fields = ['timestamp']