from django.db import models
from django.utils import timezone

## https://chatgpt.com/c/670fe509-6a2c-8006-8901-87663cdd4d08

class Product(models.Model):
    category = models.CharField(max_length=255)
    subcategory = models.CharField(max_length=255, null=True, blank=True)
    brand = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    image_url = models.URLField(max_length=1024)
    secondary_images = models.JSONField()
    product_link = models.URLField(max_length=1024)  # Link to the product page
    created_at = models.DateTimeField(auto_now_add=True)
    brand_product_id = models.CharField(max_length=255, null=True, blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    discount_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    description = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f'{self.id}'
    

class ProductEmbeddings(models.Model):
    product = models.OneToOneField(Product, on_delete=models.CASCADE, related_name='embedding')
    text_embedding = models.JSONField(null=True, blank=True)
    image_embedding = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f'{self.id} {self.product.id}'

    
class Media(models.Model):
    MEDIA_TYPE_CHOICES = [
        ('IMAGE', 'Image'),
        ('VIDEO', 'Video')
    ]

    type = models.CharField(max_length=10, choices=MEDIA_TYPE_CHOICES)
    url = models.URLField()  
    post = models.ForeignKey('Post', on_delete=models.CASCADE, related_name='media')  # Linked to 'TAGGED_POST'

    def __str__(self):
        return f"{self.type} - {self.id}"



class Post(models.Model):
    POST_TYPE_CHOICES = [
        ('PRODUCT_POST', 'PRODUCT_POST'),
        ('TAGGED_POST', 'TAGGED_POST')
    ]

    post_type = models.CharField(max_length=20, choices=POST_TYPE_CHOICES)
    title = models.CharField(max_length=255, blank=True, null=True)  
    description = models.TextField(blank=True, null=True) # Title and Description for TAGGED_POST
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True, related_name='posts') # Product for PRODUCT_POST
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.post_type} - {self.id}"
    

class TaggedProduct(models.Model):
    post = models.ForeignKey(Post, related_name='tagged_products', on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)

    def __str__(self):
        return f"Tagged {self.product} in {self.post}"