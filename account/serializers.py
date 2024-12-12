from .models import UserPreferences
from rest_framework import serializers


class UserPreferencesSerializer(serializers.ModelSerializer):
    aesthetics = serializers.ListField(
        child=serializers.CharField(max_length=50),
        default=list
    )
    avoid_styles = serializers.ListField(
        child=serializers.CharField(max_length=50),
        default=list
    )

    class Meta:
        model = UserPreferences
        fields = ['user', 'aesthetics', 'avoid_styles']