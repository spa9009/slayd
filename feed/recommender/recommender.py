from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from feed.models import Product, ProductEmbeddings

def get_combined_recommendations(product_id, user_id, num_recommendations=10):
    product = Product.objects.filter(id=product_id).first()
    product_embedding = ProductEmbeddings.objects.filter(product=product_id).first()

    if not product_embedding or not product:
        print("Image not available in the database.")
        return
    
    category_weight = 0.5
    image_weight = 0.5
    brand_boost = 1.2
    recommendations = []

    for embedding in ProductEmbeddings.objects.exclude(id=product_id):
        product_sim = Product.objects.filter(id=product_id).first()

        if not product_sim:
            continue
        elif product_sim.brand == product.brand:
            continue
        text_similarity = cosine_similarity([product_embedding.text_embedding], [np.array(embedding.text_embedding)])[0][0]
        image_similarity = cosine_similarity([product_embedding.image_embedding], [np.array(embedding.image_embedding)])[0][0]

        combined_similarity = (category_weight * text_similarity) + (image_weight * image_similarity)

        recommendations.append((product_sim, combined_similarity))
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:num_recommendations]
        recommended_products = [{"id": product.id, "name": product.name, "similarity": sim} for product, sim in recommendations]

    print(recommended_products)


get_combined_recommendations(1, 1, 5)


