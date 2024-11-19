from rest_framework import serializers
from .models import UserActivity, Follow

class UserActivitySerializer(serializers.ModelSerializer):
    class Meta: 
        model = UserActivity
        fields = ['user', 'post', 'action', 'timestamp']
    
    def validate_action(self, value):
        if value not in dict(UserActivity.ACTIONS):
            raise serializers.ValidationError('Invalid action type')
        return value
    
class FollowSerializer(serializers.ModelSerializer):
    class Meta: 
        model = Follow
        fields = ['user', 'publisher_type', 'publisher', 'timestamp']
        read_only_fields = ['timestamp']