from django.db import models
from account.models import User
from feed.models import Post

## Explore later when scaling whether this needs to be moved to NoSQL DB

class UserActivity(models.Model):
    ACTIONS = [
        ('save', 'SAVE'),
        ('like', 'LIKE'),
        ('wishlist', 'WISHLIST')
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='activity')
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='activity')
    action = models.CharField(max_length=10, choices=ACTIONS)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['user', 'post', 'action'], name='unique_user_post_action')
        ]

    def __str__(self):
        return f"{self.user} {self.action} {self.post}"
    
class Follow(models.Model):
    PUBLISHER_TYPE = [
        ('brand', 'BRAND'),
        ('influencer', 'INFLUENCER')
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='follow')
    publisher_type = models.CharField(max_length=10, choices=PUBLISHER_TYPE)
    publisher = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['user', 'publisher'], name='unique_user_publisher_action')
        ]
    
    def __str__(self):
        return f"{self.user} follows {self.publisher}"
