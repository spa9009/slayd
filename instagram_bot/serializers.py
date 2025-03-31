from rest_framework import serializers
from .models import VideoPost, ChildImage

class VideoPostSerializer(serializers.ModelSerializer):
    child_images = serializers.SerializerMethodField()
    
    class Meta:
        model = VideoPost
        fields = ['id', 'video_url', 'created_at', 'child_images']
    
    def get_child_images(self, obj):
        return [image.image_url for image in obj.images.all()]