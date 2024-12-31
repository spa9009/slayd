from feed.models import Media, Product
from django.db import transaction
from django.core.management.base import BaseCommand

s3_prefix = "https://feed-images-01.s3.ap-south-1.amazonaws.com/"
cdn_prefix = "https://d19dlu1w9mnmln.cloudfront.net/"

class Command(BaseCommand):
    def handle(self, *args, **options):
        ## Transform s3 links for each product to cdn link: 
        products = Product.objects.all()

        for product in products: 
            isUpdated = False
            if product.image_url.startswith(s3_prefix):
                product.image_url = product.image_url.replace(s3_prefix, cdn_prefix)
                isUpdated = True

            secondary_images = []
            if isinstance(product.secondary_images, list):
                for image in product.secondary_images:
                    if image is not None and image.startswith(s3_prefix):
                        secondary_images.append(image.replace(s3_prefix, cdn_prefix))
                        isUpdated = True
            
            if isUpdated:
                if product.id % 10 == 0:
                    print(f"ProductId: {product.id}")
                product.secondary_images = secondary_images
                product.save()

        print("Changed all Products")

        medias = Media.objects.all()
        for media in medias: 
            if media.url.startswith(s3_prefix):
                media.url = media.url.replace(s3_prefix, cdn_prefix)
                media.save()

        print("Changed all Media")


        


