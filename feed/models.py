from django.db import models

class Product(models.Model):
    product_id = models.CharField(max_length=255, unique=True)
    category = models.CharField(max_length=255)
    subcategory = models.CharField(max_length=255)
    brand = models.CharField(max_length=255)
    product_name = models.CharField(max_length=255)
    product_image_url = models.URLField(max_length=1024)  # S3 or external storage URL
    product_link = models.URLField(max_length=1024)  # Link to the product page
    created_at = models.DateTimeField(auto_now_add=True)
    brand_product_id = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)

    def __str__(self):
        return self.product_id