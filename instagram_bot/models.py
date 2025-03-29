from django.db import models


class VideoPost(models.Model):
    video_url = models.URLField(max_length=2000)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Video Post {self.id}"

class ChildImage(models.Model):
    video_post = models.ForeignKey(
        VideoPost,
        on_delete=models.CASCADE,
        related_name='images'
    )
    image_url = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Child Image {self.id} for Video {self.video_post_id}"