from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Product, Post

@receiver(post_save, sender=Product)
def create_product_post(sender, instance, created, **kwargs):
    if created:
        Post.objects.create(
            post_type='PRODUCT_POST',
            product=instance
        )