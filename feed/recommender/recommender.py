from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from feed.models import Post, Product

# def get_combined_recommendations(product_id, user_id, num_recommendations=10)