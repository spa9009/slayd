from .models import UserPreferences
from rest_framework import serializers


class UserProfileSerializer(serializers.ModelSerializer):
    aesthetics = serializers.ListField(
        child=serializers.CharField(max_length=50),
        default=list
    )
    avoided_styles = serializers.ListField(
        child=serializers.CharField(max_length=50),
        default=list
    )

    class Meta:
        model = UserPreferences
        fields = ['user', 'aesthetics', 'avoided_styles']