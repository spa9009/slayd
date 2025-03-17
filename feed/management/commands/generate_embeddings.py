from django.core.management.base import BaseCommand
from feed.models import MyntraProducts, MyntraProductEmbeddings
import numpy as np
from PIL import Image
from io import BytesIO
from fashion_clip.fashion_clip import FashionCLIP
from django.db import transaction
import requests
from math import ceil

class Command(BaseCommand):
    help = 'Extract embeddings for products in batches and save them to the database'

    def handle(self, *args, **kwargs):
        fclip = FashionCLIP('fashion-clip')
        self.stdout.write("Model loaded. Proceeding further")

        products = list(MyntraProducts.objects.filter(
            embedding__isnull=True  # Only select products without embeddings
        ) | MyntraProducts.objects.filter(
            embedding__image_embedding__isnull=True  # Retry products with missing image embeddings
        ) | MyntraProducts.objects.filter(
            embedding__text_embedding__isnull=True  # Retry products with missing text embeddings
        ))
        print("Total unprocessed products are: " + str(len(products)))
        batch_size = 64
        total_batches = ceil(len(products) / batch_size)

        for batch_index in range(total_batches):
            start_index = batch_index * batch_size
            end_index = min(start_index + batch_size, len(products))
            
            batch_products = products[start_index:end_index]

            images = {}
            categories = {}
            brands = {}

            for product in batch_products:
                image_url = product.image_url.replace('https://feed-images-01.s3.ap-south-1.amazonaws.com', 'https://d19dlu1w9mnmln.cloudfront.net')
                try:
                    response = requests.get(image_url, timeout=10)
                    img = Image.open(BytesIO(response.content)).convert("RGB").resize((224, 224))
                    images[product.id] = img
                    categories[product.id] = f"{product.category} {product.subcategory} {product.gender} {product.name} {product.color}"
                    brands[product.id] = product.brand
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Failed to process product {product.id}: {str(e)}"))

            if images:
                self.stdout.write(f"Processing batch {batch_index + 1}/{total_batches} with {len(images)} products")
                try:
                    image_embeddings_dict, text_embeddings_dict = convert_embeddings(
                        images=images, categories=categories, brands=brands, model=fclip
                    )

                    with transaction.atomic():
                        for product_id in images.keys():
                            try:
                                product = MyntraProducts.objects.get(id=product_id)

                                MyntraProductEmbeddings.objects.update_or_create(
                                    myntra_product=product,
                                    defaults={
                                        'image_embedding': image_embeddings_dict[product_id].tolist(),
                                        'text_embedding': text_embeddings_dict[product_id].tolist(),
                                    }
                                )

                            except MyntraProducts.DoesNotExist:
                                self.stdout.write(f"Product with ID {product_id} not found.")

                    self.stdout.write(f"Batch {batch_index + 1}/{total_batches} processed successfully")
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Failed to process embeddings for batch {batch_index + 1}: {str(e)}"))
            else:
                self.stdout.write(self.style.WARNING(f"No valid images in batch {batch_index + 1}"))

        self.stdout.write("All batches processed.")

def convert_embeddings(images, categories, brands, model):
    image_embeddings = model.encode_images(images=list(images.values()), batch_size=32)
    category_embeddings = model.encode_text(text=list(categories.values()), batch_size=32)
    brand_embeddings = model.encode_text(text=list(brands.values()), batch_size=32)

    assert len(images.keys()) == len(image_embeddings)
    assert len(categories.keys()) == len(category_embeddings)
    assert len(brands.keys()) == len(brand_embeddings)

    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
    category_embeddings = category_embeddings / np.linalg.norm(category_embeddings, ord=2, axis=-1, keepdims=True)
    brand_embeddings = brand_embeddings / np.linalg.norm(brand_embeddings, ord=2, axis=-1, keepdims=True)

    text_embeddings = category_embeddings
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)

    image_embeddings_dict = {product_id: embedding for product_id, embedding in zip(images.keys(), image_embeddings)}
    text_embeddings_dict = {product_id: embedding for product_id, embedding in zip(images.keys(), text_embeddings)}

    return image_embeddings_dict, text_embeddings_dict