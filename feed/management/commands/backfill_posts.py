from django.core.management.base import BaseCommand
from feed.models import Product, Post 

class Command(BaseCommand):
    help = 'Backfill posts by linking products to posts'

    def handle(self, *args, **kwargs):
        products = Product.objects.all()  

        for product in products:
            if not Post.objects.filter(product=product).exists():
                post = Post.objects.create(
                    post_type = 'PRODUCT_POST',
                    product=product
                )
                self.stdout.write(self.style.SUCCESS(f"Created post for product {product.product_name}"))

        self.stdout.write(self.style.SUCCESS("Backfill complete!"))