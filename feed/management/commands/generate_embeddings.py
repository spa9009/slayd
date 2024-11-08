from django.core.management.base import BaseCommand
from feed.models import Product, ProductEmbeddings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import boto3
from PIL import Image
from io import BytesIO


s3_client = boto3.client(
    's3',
    aws_access_key_id='', ## add secret
    aws_secret_access_key='', ## add secret
    region_name='' ## add secret
)


class Command(BaseCommand):
    help = 'Extract embeddings for products and save them to the database'

    def handle(self, *args, **kwargs):
        base_model = ResNet50(weights='imagenet')

        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        self.stdout.write("Model loaded. Proceeding further")
        products = Product.objects.all()
        text_data = [f"{p.category} {p.subcategory}" for p in products]
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data).toarray()

        for idx, product in enumerate(products):
            image_url = product.image_url
            try :
                product_embedding, created = ProductEmbeddings.objects.get_or_create(product=product)

                if not created: 
                    continue

                bucket_name, key = parse_s3_url(image_url)
                self.stdout.write(f"Bucket name is {bucket_name}")  
                response = s3_client.get_object(Bucket=bucket_name, Key=key)
                img_data = response['Body'].read()

                img = Image.open(BytesIO(img_data)).resize((224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                embedding = model.predict(x).flatten()

                if idx == 1:
                    self.stdout.write(f"image embeddings: {embedding.tolist()}")  
                product_embedding, created = ProductEmbeddings.objects.get_or_create(product=product)
                product_embedding.text_embedding = tfidf_matrix[idx].tolist()
                product_embedding.image_embedding = embedding.tolist()

                product_embedding.save()
                self.stdout.write(self.style.SUCCESS(f"Processed product {product.id}"))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Failed to process product {product.id}: {str(e)}"))

def parse_s3_url(s3_url):
    """Helper function to parse an S3 URL into bucket and key."""
    s3_parts = s3_url.replace("https://", "").split("/", 1)
    bucket_name = s3_parts[0].split(".s3")[0]
    key = s3_parts[1]
    return bucket_name, key