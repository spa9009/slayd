import os
# Remove the offline mode settings since they're preventing the model download
import numpy as np
import faiss
from PIL import Image
import requests
from io import BytesIO
from feed.models import MyntraProducts
from feed.apps import fclip, clip_model, clip_processor
from django.conf import settings
import psutil
import logging
import torch
import traceback

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
            
            # Initialize models
            logging.debug("Initializing models...")
            try:
                self.fclip = fclip
                self.model = clip_model
                self.processor = clip_processor
                logging.debug("Models initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize models: {str(e)}")
                raise
            
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
        logger = logging.getLogger(__name__)
        try:
            logger.debug(f"Attempting to load image from: {image_path}")
            
            if image_path.startswith(('http://', 'https://')):
                try:
                    response = requests.get(image_path, timeout=10)
                    response.raise_for_status()  # Raise an exception for bad status codes
                    logger.debug(f"Image download status code: {response.status_code}")
                    logger.debug(f"Image content type: {response.headers.get('content-type')}")
                    logger.debug(f"Image size: {len(response.content)} bytes")
                    
                    image = Image.open(BytesIO(response.content))
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to download image: {str(e)}")
                    raise
                except Exception as e:
                    logger.error(f"Failed to open downloaded image: {str(e)}")
                    raise
            else:
                image = Image.open(image_path)
            
            # Check image mode and convert if necessary
            logger.debug(f"Original image mode: {image.mode}")
            logger.debug(f"Original image size: {image.size}")
            
            if image.mode != "RGB":
                image = image.convert("RGB")
                logger.debug("Converted image to RGB mode")
            
            # Resize image
            resized_image = image.resize((224, 224))
            logger.debug("Image resized to 224x224")
            
            return resized_image
            
        except Exception as e:
            logger.error(f"Error in load_and_process_image: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to process image: {str(e)}")

    def get_text_description(self, image):
        """Generate detailed text description using CLIP zero-shot classification"""
        apparel_types = [
            "dress", "kurta", "shirt", "pants", "skirt", 
            "saree", "t-shirt", "jacket"
        ]
        
        # Generate category lists
        apparel_categories = [f"a photo of a {x}" for x in apparel_types]
        
        # Use CLIP for classification
        inputs = self.processor(
            text=apparel_categories,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)
            predicted_apparel_idx = probs[0].argmax().item()
        
        predicted_apparel = apparel_types[predicted_apparel_idx]

        # If it's a dress, get more detailed attributes using DressClassifier
        if predicted_apparel == "dress":
            dress_classifier = DressClassifier(self.model, self.processor)
            detailed_description = dress_classifier.generate_description(image)
            return detailed_description
        else:
            # For non-dress items, use basic color classification
            dress_classifier = DressClassifier(self.model, self.processor)
            color_categories = [
                f"a photo of {'an' if x[0].lower() in 'aeiou' else 'a'} {x} colored clothing" 
                for x in dress_classifier.dress_colors  # Use dress_classifier's colors
            ]
            
            inputs = self.processor(
                text=color_categories,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1)
                predicted_color_idx = probs[0].argmax().item()
            
            predicted_color = dress_classifier.dress_colors[predicted_color_idx]
            return f"This is a {predicted_color.lower()} {predicted_apparel}"

    def get_similar_products(self, image_path, top_k=20, search_type='combined_75'):
        logger = logging.getLogger(__name__)
        try:
            logger.debug(f"Starting similarity search for {image_path}")
            
            # Process input image with detailed error handling
            try:
                image = self.load_and_process_image(image_path)
                logger.debug("Image loaded and processed successfully")
            except Exception as e:
                logger.error(f"Failed to load/process image: {str(e)}")
                raise RuntimeError(f"Image processing failed: {str(e)}")

            # Get image embedding with error handling
            try:
                logger.debug("Generating image embeddings")
                image_embeddings = self.fclip.encode_images(images=[image], batch_size=1)
                image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
                logger.debug("Image embeddings generated successfully")
            except Exception as e:
                logger.error(f"Failed to generate image embeddings: {str(e)}")
                raise RuntimeError(f"Embedding generation failed: {str(e)}")

            # Get text description with error handling
            try:
                logger.debug("Generating text description")
                text_description = self.get_text_description(image)
                logger.debug(f"Generated text description: {text_description}")
            except Exception as e:
                logger.error(f"Failed to generate text description: {str(e)}")
                raise RuntimeError(f"Text description generation failed: {str(e)}")

            # Prepare query embedding based on search type
            if search_type == 'image':
                query_embedding = image_embeddings
            elif search_type == 'text':
                query_embedding = self.fclip.encode_text([text_description], batch_size=1)
                query_embedding = query_embedding / np.linalg.norm(query_embedding, ord=2, axis=-1, keepdims=True)
            elif search_type == 'combined_60':
                query_embedding = 0.60 * image_embeddings + 0.40 * self.fclip.encode_text([text_description], batch_size=1)
                query_embedding = query_embedding / np.linalg.norm(query_embedding, ord=2, axis=-1, keepdims=True)
            elif search_type == 'combined_75':
                query_embedding = 0.75 * image_embeddings + 0.25 * self.fclip.encode_text([text_description], batch_size=1)
                query_embedding = query_embedding / np.linalg.norm(query_embedding, ord=2, axis=-1, keepdims=True)
            elif search_type == 'concat':
                query_embedding = np.concatenate([image_embeddings[0], self.fclip.encode_text([text_description], batch_size=1)[0]])
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
                    'product_name': product.name,
                    'product_price': product.price,
                    'discount_price': product.discount_price,
                    'product_image': product.image_url,
                    'product_brand': product.brand,
                    'product_marketplace': product.marketplace,
                    'similarity_score': float(distances[0][i]),
                    'description': text_description if i == 0 else None  # Include description for first result only
                })

            return similar_products

        except Exception as e:
            logger.error(f"Error in get_similar_products: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def _monitor_memory_usage(self):
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        
        if memory_usage_mb > 1000:  # Alert if using more than 1GB
            logging.warning(f"High memory usage detected: {memory_usage_mb:.2f} MB") 


class DressClassifier:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        
        self.dress_lengths = [
            "Micro", "Mini", "Knee-Length", "Midi", "Maxi",
            "High-Low", "Tea-Length", "Floor-Length"
        ]
        
        self.dress_necklines = [
            "V-Neck", "Round Neck", "Square Neck", "Halter Neck",
            "Off-Shoulder", "Boat Neck", "Sweetheart", "High Neck",
            "Plunge Neck", "One-Shoulder", "Cowl Neck", "Collared", "Strapless"
        ]
        
        self.dress_sleeves = [
            "Sleeveless", "Cap Sleeves", "Short Sleeves", "3/4 Sleeves",
            "Full Sleeves", "Bell Sleeves", "Puff Sleeves", "Cold Shoulder",
            "Bishop Sleeves", "Kimono Sleeves", "Ruffle Sleeves"
        ]
        
        self.dress_materials = [
            "Cotton", "Linen", "Chiffon", "Satin", "Velvet", "Denim",
            "Lace", "Polyester", "Georgette", "Silk", "Sequin", "Tulle",
            "Crepe", "Organza", "Leather", "Knits", "Jacquard", "Tweed"
        ]
        
        self.dress_fits = [
            "Bodycon", "A-line", "Fit & Flare", "Shift", "Wrap",
            "Empire Waist", "Peplum", "Smocked", "Mermaid", "Straight Cut"
        ]
        
        self.dress_design_features = [
            "Slit", "Cutout", "Ruffles", "Fringe", "Ruched",
            "Embellished", "Backless", "Corset Detail", "Asymmetrical Hem",
            "Tiered Layers", "Bow Detail", "Lace-Up", "Belted"
        ]

        # Add colors list to DressClassifier
        self.dress_colors = [
            "Coffee Brown", "Orange", "Red", "Green", "Grey", "Pink", 
            "Blue", "Purple", "White", "Black", "Yellow", "Beige",
            "Maroon", "Burgundy", "Brown"
        ]

    def zero_shot_classification(self, image, categories):
        """Perform zero-shot classification using CLIP"""
        inputs = self.processor(
            text=categories,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            self.model = self.model.cuda()

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)
            
        # Get the category with highest probability
        predicted_idx = probs[0].argmax().item()
        return categories[predicted_idx]

    def generate_description(self, image):
        # Color classification
        color_categories = [
            f"a photo of {'an' if color[0].lower() in 'aeiou' else 'a'} {color} colored dress" 
            for color in self.dress_colors
        ]
        color_pred = self.zero_shot_classification(image, color_categories)
        predicted_color = self.dress_colors[color_categories.index(color_pred)]

        # Length classification
        length_categories = [f"a photo of a {length.lower()} dress" for length in self.dress_lengths]
        length_pred = self.zero_shot_classification(image, length_categories)
        predicted_length = self.dress_lengths[length_categories.index(length_pred)]

        # Fit classification
        fit_categories = [f"a photo of a {fit.lower()} dress" for fit in self.dress_fits]
        fit_pred = self.zero_shot_classification(image, fit_categories)
        predicted_fit = self.dress_fits[fit_categories.index(fit_pred)]

        # Neckline classification
        neckline_categories = [f"a photo of a dress with {neckline.lower()}" for neckline in self.dress_necklines]
        neckline_pred = self.zero_shot_classification(image, neckline_categories)
        predicted_neckline = self.dress_necklines[neckline_categories.index(neckline_pred)]

        # Sleeve classification
        sleeve_categories = [f"a photo of a {sleeve.lower()} dress" for sleeve in self.dress_sleeves]
        sleeve_pred = self.zero_shot_classification(image, sleeve_categories)
        predicted_sleeve = self.dress_sleeves[sleeve_categories.index(sleeve_pred)]

        # Material classification
        material_categories = [f"a photo of a {material.lower()} dress" for material in self.dress_materials]
        material_pred = self.zero_shot_classification(image, material_categories)
        predicted_material = self.dress_materials[material_categories.index(material_pred)]

        # Design feature classification
        feature_categories = [f"a photo of a dress with {feature.lower()}" for feature in self.dress_design_features]
        feature_pred = self.zero_shot_classification(image, feature_categories)
        predicted_feature = self.dress_design_features[feature_categories.index(feature_pred)]

        # Build description
        description_parts = [
            predicted_color.lower(),
            predicted_material.lower(),
            predicted_length.lower(),
            predicted_fit.lower(),
            predicted_neckline.lower(),
            predicted_sleeve.lower()
        ]
        
        base_description = "This is a " + " ".join(description_parts) + " dress"
        base_description += f" with {predicted_feature.lower()}"

        return base_description
