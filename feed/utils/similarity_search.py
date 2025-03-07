import os
import numpy as np
import faiss
from PIL import Image
import requests
from io import BytesIO
from feed.models import MyntraProducts
from django.conf import settings
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor

class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class SimilaritySearcher(metaclass=SingletonMeta):
    def __init__(self):
        self._is_initialized = False
        self._initialize()  # Call initialize in __init__
    
    def _initialize(self):
        if self._is_initialized:
            return
            
        try:
            logging.debug("Starting initialization...")
            if settings.DEBUG:
                base_path = os.path.join(settings.BASE_DIR, 'indices')
            else:
                base_path = getattr(settings, 'FAISS_INDICES_PATH', '/opt/fashion_recommendation/indices')
            
            logging.debug(f"Using base path: {base_path}")
            
            # Initialize FashionCLIP
            logging.debug("Checking FashionCLIP model...")
            if settings.FASHIONCLIP_MODEL is None:
                raise RuntimeError("FashionCLIP model was not properly initialized at startup")
            
            self.fclip = settings.FASHIONCLIP_MODEL
            logging.debug(f"FashionCLIP model accessed successfully")
            
            # Load all FAISS indices
            logging.debug("Loading FAISS indices...")
            try:
                self.indices = {
                    'image': self._load_index(os.path.join(base_path, 'image_faiss_index_fclip.bin')),
                    'text': self._load_index(os.path.join(base_path, 'text_faiss_index_fclip.bin')),
                    'combined_60': self._load_index(os.path.join(base_path, 'combined_faiss_index_60_fclip.bin')),
                    'combined_75': self._load_index(os.path.join(base_path, 'combined_faiss_index_75_fclip.bin')),
                    'concat': self._load_index(os.path.join(base_path, 'concat_faiss_index_fclip.bin'))
                }
                logging.debug("Loaded all indices successfully")
                
                self.product_ids = self._load_product_ids(os.path.join(base_path, 'product_ids.npy'))
                logging.debug("Loaded product IDs")
            except Exception as e:
                logging.error(f"Failed to load FAISS indices: {str(e)}")
                raise
            
            self._is_initialized = True
            logging.debug("Initialization completed successfully")
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize SimilaritySearcher: {str(e)}")

    def _load_index(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index not found at {path}")
        return faiss.read_index(path)

    def _load_product_ids(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Product IDs file not found at {path}")
        return np.load(path)

    def load_and_process_image(self, image_path):
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        return image.resize((224, 224))

    def get_text_description(self, image):
        """Generate detailed text description using FashionCLIP zero-shot classification"""
        apparel_types = [
            "dress", "kurta", "shirt", "pants", "skirt", 
            "saree", "t-shirt", "jacket"
        ]
        colors = [
            "Orange", "Red", "Green", "Grey", "Pink", 
            "Blue", "Purple", "White", "Black", "Yellow", "Beige",
            "Maroon", "Burgundy", "Brown"
        ]

        # Generate category lists
        apparel_categories = [f"a photo of a {x}" for x in apparel_types]
        color_categories = [f"a photo of {'an' if x[0].lower() in 'aeiou' else 'a'} {x} colored clothing" for x in colors]

        # Get apparel type using zero-shot classification
        predicted_apparel_idx = self.fclip.zero_shot_classification([image], apparel_categories)[0]
        predicted_apparel = apparel_types[apparel_categories.index(predicted_apparel_idx)]

        # If it's a dress, get more detailed attributes
        if predicted_apparel == "dress":
            dress_classifier = DressClassifier(self.fclip)  # Pass fclip instance
            detailed_description = dress_classifier.generate_description(image)
            return detailed_description
        else:
            # Get color using zero-shot classification
            predicted_color_idx = self.fclip.zero_shot_classification([image], color_categories)[0]
            predicted_color = colors[color_categories.index(predicted_color_idx)]
            return f"This is a {predicted_color.lower()} {predicted_apparel}"

    def get_similar_products(self, image_path, top_k=20, search_type='combined_75'):
        """
        Get similar products using specified search type.
        
        Args:
            image_path: Path or URL to the query image
            top_k: Number of similar products to return
            search_type: One of 'image', 'text', 'combined_60', 'combined_75', or 'concat'
        """
        if not self._is_initialized:
            self._initialize()
            
        try:
            logging.debug(f"Processing request for image: {image_path}")
            
            # Process input image
            image = self.load_and_process_image(image_path)
            
            # Get image embedding
            image_embeddings = self.fclip.encode_images(images=[image], batch_size=1)
            image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)

            # Prepare query embedding based on search type
            if search_type == 'image':
                query_embedding = image_embeddings
            else:
                # Get text description and convert dictionary to string
                description_dict = self.get_text_description(image)
                if isinstance(description_dict, dict):
                    text_description = f"This is a {description_dict['color']} {description_dict['length']} {description_dict['fit']} dress with {description_dict['neckline']}, {description_dict['sleeve']}, made of {description_dict['material']}, featuring {description_dict['feature']}"
                else:
                    text_description = description_dict  # In case it's already a string

                # Get text embedding
                text_embeddings = self.fclip.encode_text([text_description], batch_size=1)
                text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)
                if search_type == 'text':
                    query_embedding = text_embeddings
                elif search_type == 'combined_60':
                    query_embedding = 0.60 * image_embeddings + 0.40 * text_embeddings
                    query_embedding = query_embedding / np.linalg.norm(query_embedding, ord=2, axis=-1, keepdims=True)
                elif search_type == 'combined_75':
                    query_embedding = 0.75 * image_embeddings + 0.25 * text_embeddings
                    query_embedding = query_embedding / np.linalg.norm(query_embedding, ord=2, axis=-1, keepdims=True)
                elif search_type == 'concat':
                    query_embedding = np.concatenate([image_embeddings[0], text_embeddings[0]])
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)
                    query_embedding = query_embedding.reshape(1, -1)
                else:
                    raise ValueError(f"Invalid search type: {search_type}")

            # Perform search using appropriate index
            distances, idx = self.indices[search_type].search(
                np.array(query_embedding, dtype="float32"),
                top_k
            )

            # Get product details
            similar_products = []
            for i in range(len(idx[0])):
                product_id = str(self.product_ids[idx[0][i]])
                product = MyntraProducts.objects.get(id=product_id)
                similar_products.append({
                    'id': product.id,
                    'product_link': product.product_link,
                    'similarity_score': float(distances[0][i]),
                })

            return similar_products

        except Exception as e:
            logging.error(f"Error in similarity search: {str(e)}", exc_info=True)
            raise Exception(f"Error in similarity search: {str(e)}")


class DressClassifier:
    def __init__(self, fclip):
        self.fclip = fclip
        
        self.dress_lengths = [
            "Mini", "Knee-Length", "Midi", "Maxi", "Floor-Length"
        ]

        self.dress_necklines = [
            "V-Neck", "Round Neck", "Square Neck", 
            "Off-Shoulder", "Halter Neck", "Sweetheart"
        ]

        self.dress_sleeves = [
            "Sleeveless", "Short Sleeves", "Full Sleeves", 
            "Puff Sleeves", "Cold Shoulder"
        ]

        self.dress_materials = [
            "Cotton", "Satin", "Velvet", "Denim", 
            "Lace", "Silk", "Polyester", "Sequin"
        ]

        self.dress_fits = [
            "Bodycon", "A-line", "Fit & Flare", "Wrap"
        ]

        self.dress_design_features = [
            "Slit", "Cutout", "Ruffles", "Backless", "Asymmetrical Hem"
        ]

        self.dress_colors = [
            "Orange", "Red", "Green", "Grey", "Pink", 
            "Blue", "Purple", "White", "Black", "Yellow", "Beige", 
            "Maroon", "Burgundy", "Brown"
        ]

    def classify_attribute(self, image, categories, attribute_list):
        pred = self.fclip.zero_shot_classification([image], categories)[0]
        return attribute_list[categories.index(pred)]

    def generate_description(self, image):
        # Prepare all categories in batches
        results = {}
        
        # Color classification
        color_categories = [f"a photo of {'an' if color[0].lower() in 'aeiou' else 'a'} {color} colored dress" 
                           for color in self.dress_colors]
        color_pred = self.fclip.zero_shot_classification([image], color_categories)[0]
        results['color'] = self.dress_colors[color_categories.index(color_pred)]
        
        # Length classification
        length_categories = [f"a photo of a {length.lower()} dress" for length in self.dress_lengths]
        length_pred = self.fclip.zero_shot_classification([image], length_categories)[0]
        results['length'] = self.dress_lengths[length_categories.index(length_pred)]
        
        # Fit classification
        fit_categories = [f"a photo of a {fit.lower()} dress" for fit in self.dress_fits]
        fit_pred = self.fclip.zero_shot_classification([image], fit_categories)[0]
        results['fit'] = self.dress_fits[fit_categories.index(fit_pred)]
        
        # Neckline classification
        neckline_categories = [f"a photo of a dress with {neckline.lower()}" for neckline in self.dress_necklines]
        neckline_pred = self.fclip.zero_shot_classification([image], neckline_categories)[0]
        results['neckline'] = self.dress_necklines[neckline_categories.index(neckline_pred)]
        
        # Sleeve classification
        sleeve_categories = [f"a photo of a {sleeve.lower()} dress" for sleeve in self.dress_sleeves]
        sleeve_pred = self.fclip.zero_shot_classification([image], sleeve_categories)[0]
        results['sleeve'] = self.dress_sleeves[sleeve_categories.index(sleeve_pred)]
        
        # Material classification
        material_categories = [f"a photo of a {material.lower()} dress" for material in self.dress_materials]
        material_pred = self.fclip.zero_shot_classification([image], material_categories)[0]
        results['material'] = self.dress_materials[material_categories.index(material_pred)]
        
        # Feature classification
        feature_categories = [f"a photo of a dress with {feature.lower()}" for feature in self.dress_design_features]
        feature_pred = self.fclip.zero_shot_classification([image], feature_categories)[0]
        results['feature'] = self.dress_design_features[feature_categories.index(feature_pred)]
        
        return results
