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
    gender = models.CharField(max_length=10)

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


class MyntraProducts(models.Model):
    category = models.CharField(max_length=255)
    subcategory = models.CharField(max_length=255, null=True, blank=True)
    brand = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    image_url = models.URLField(max_length=1024)
    secondary_images = models.JSONField(null=True, blank=True)
    product_link = models.URLField(max_length=1024)  # Link to the product page
    created_at = models.DateTimeField(auto_now_add=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    discount_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    description = models.CharField(max_length=255, null=True, blank=True)
    gender = models.CharField(max_length=10)
    color = models.CharField(max_length=255, null=True, blank=True)
    marketplace = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f'{self.id}'
     
class MyntraProductEmbeddings(models.Model):
    myntra_product = models.OneToOneField(MyntraProducts, on_delete=models.CASCADE, related_name='embedding')
    text_embedding = models.JSONField(null=True, blank=True)
    image_embedding = models.JSONField(null=True, blank=True)
    fclip_text_embedding = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f'{self.id} {self.myntra_product.id}'

class MyntraProductTags(models.Model):
    myntra_product = models.OneToOneField(MyntraProducts, on_delete=models.CASCADE, related_name='tag')
    color = models.CharField(max_length=255, null=True, blank=True)
    fit = models.CharField(max_length=255, null=True, blank=True)
    length = models.CharField(max_length=255, null=True, blank=True)
    print = models.CharField(max_length=255, null=True, blank=True)
    sleeve = models.CharField(max_length=255, null=True, blank=True)
    material = models.CharField(max_length=255, null=True, blank=True)
    neckline = models.CharField(max_length=255, null=True, blank=True)
    design = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f'{self.id} {self.myntra_product.id}'
    

class Curation(models.Model):
    title = models.CharField(max_length=255, null=True, blank=True)
    curation_image = models.URLField()
    products = models.ManyToManyField(MyntraProducts, related_name='curations')
    created_at = models.DateTimeField(default=timezone.now)
    related_curations = models.ManyToManyField('self', blank=True, symmetrical=False, related_name='related_to')

    def __str__(self):
        return f"{self.title}" 

class PostInteraction(models.Model):
    senderId = models.CharField(max_length=255)
    media_url = models.URLField(max_length=1024)
    media_type = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.id} {self.senderId}'

class DetectedObjectProducts(models.Model):
    """
    Model to store product IDs for each detected object in an image.
    Each detected object (label with bounding box) has its own set of similar products.
    When is_whole_image=True, the record represents the entire image, not a specific object.
    """
    image_url = models.URLField(max_length=1024)
    label = models.CharField(max_length=255)
    # Bounding box coordinates (normalized 0-1)
    x = models.FloatField()
    y = models.FloatField()
    width = models.FloatField()
    height = models.FloatField()
    confidence = models.FloatField()
    similar_products = models.JSONField()
    is_whole_image = models.BooleanField(default=False)
    vision_result = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['image_url']),
            models.Index(fields=['label']),
            models.Index(fields=['is_whole_image']),
        ]
        verbose_name = "Detected Object Products"
        verbose_name_plural = "Detected Object Products"
    
    def __str__(self):
        if self.is_whole_image:
            return f'Whole image: {self.image_url}'
        return f'{self.label} at ({self.x:.2f}, {self.y:.2f}) in {self.image_url}'