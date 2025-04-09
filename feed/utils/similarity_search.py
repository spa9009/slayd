import os
import numpy as np
import faiss
from PIL import Image
import requests
from io import BytesIO
from feed.models import MyntraProducts
from feed.apps import get_model_instance
from django.conf import settings
import psutil
import logging
import torch
import traceback
import gc
from contextlib import contextmanager
import hashlib
from django.core.cache import cache
from threading import Lock
from feed.utils.classifier.dress_classifier import DressClassifier
from feed.utils.classifier.tops_classifier import TopsClassifier
from feed.utils.classifier.jeans_classifier import JeansClassifier
from feed.utils.classifier.skirt_classifier import SkirtClassifier
from feed.utils.classifier.pant_classifier import PantClassifier

class SingletonMeta(type):
    _instances = {}
    _lock = Lock()
    
    def __call__(cls, *args, **kwargs):
        with cls._lock:
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
                base_path = os.path.join(settings.BASE_DIR, 'indices_ivf')
            else:
                base_path = getattr(settings, 'FAISS_INDICES_PATH', '/opt/fashion_recommendation/indices')
            
            logging.debug(f"Using base path: {base_path}")
            
            # Initialize models
            logging.debug("Initializing models...")
            try:
                model_instance = get_model_instance()
                self.model = model_instance.clip_model
                self.processor = model_instance.clip_processor
                logging.debug("Models initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize models: {str(e)}")
                raise
            
            # Load all FAISS indices
            logging.debug("Loading FAISS indices...")
            try:
                self.indices = {
                    'combined_75': self._load_index(os.path.join(base_path, 'combined_faiss_index_75_ivf50.bin')),
                }
                # Set nprobe parameter for IVF index
                self.indices['combined_75'].nprobe = 50
                logging.debug("Loaded all indices successfully")
                
                self.product_ids = self._load_product_ids(os.path.join(base_path, 'product_ids.npy'))
                logging.debug("Loaded product IDs")
            except Exception as e:
                logging.error(f"Failed to load FAISS indices: {str(e)}")
                raise
            
            self._is_initialized = True
            logging.debug("Initialization completed successfully")
            
            # Fix: Add proper cleanup and tensor management
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._cached_tensors = {}

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
            if image_path.startswith(('http://', 'https://')):
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(image_path, headers=headers, timeout=5, stream=True)
                response.raise_for_status()
                with Image.open(BytesIO(response.content)) as image:
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return image.resize((224, 224))
            else:
                with Image.open(image_path) as image:
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return image.resize((224, 224))
        except Exception as e:
            logger.error(f"Error in load_and_process_image: {str(e)}")
            raise

    def get_text_description(self, image):
        """Generate detailed text description using CLIP zero-shot classification"""
        apparel_types = [
            "dress", "kurta", "top", "shirt dress", "pants", "skirt", "suit", "jeans", "shorts",
            "Indian Ethnic", "jacket", "Blazer", "lehenga", "skorts", "sweatshirt", "hoodie",
            "jumpsuit", "bralette"
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
        if predicted_apparel == "dress" or predicted_apparel == "shirt dress":
            dress_classifier = DressClassifier(self.model, self.processor, predicted_apparel)
            detailed_description = dress_classifier.generate_description(image)
            return detailed_description
        elif predicted_apparel == "top" or predicted_apparel == "bralette":
            top_classifier = TopsClassifier(self.model, self.processor, predicted_apparel)
            detailed_description = top_classifier.generate_description(image)
            return detailed_description
        elif predicted_apparel == "jeans":
            jeans_classifier = JeansClassifier(self.model, self.processor, predicted_apparel)
            detailed_description = jeans_classifier.generate_description(image)
            return detailed_description
        elif predicted_apparel == "pants" or predicted_apparel == "shorts":
            pant_classifier = PantClassifier(self.model, self.processor, "pants")
            detailed_description = pant_classifier.generate_description(image)
            return detailed_description
        elif predicted_apparel == "skirt" or predicted_apparel == "mini skirt" or predicted_apparel == "denim skirt":
            skirt_classifier = SkirtClassifier(self.model, self.processor, "skirt")
            detailed_description = skirt_classifier.generate_description(image)
            return detailed_description
        elif predicted_apparel == "Indian Ethnic":
            return self.get_basic_description(image, predicted_apparel, False)
        else:
            return self.get_basic_description(image, predicted_apparel, True)

    def get_basic_description(self, image, predicted_apparel, should_use_apparel=True):
        """
        Generate a basic description including print pattern and color for unclassified apparel types.
        
        Args:
            image: The input image to classify
            predicted_apparel: The type of apparel detected
            should_use_apparel: Whether to include apparel type in description
            
        Returns:
            str: A description including print pattern or color and apparel type
        """
        # Print-related attributes
        print_types = [
            "Floral", "Geometric", "Abstract", "Animal", "Striped", 
            "Polka Dot", "Paisley", "Solid"
        ]
        print_color_styles = ["Monochrome", "Multicolor"]
        print_sizes = ["Small", "Medium", "Large"]
        print_status = ["Printed", "Solid"]
        
        # Colors for solid items
        dress_colors = [
            "Orange", "Red", "Green", "Grey", "Pink", 
            "Blue", "Purple", "White", "Black", "Yellow", "Beige",
            "Maroon", "Burgundy", "Brown"
        ]

        # First determine if the item is printed or solid
        print_status_categories = [
            f"a photo of a {status.lower()} {predicted_apparel}" 
            for status in print_status
        ]
        
        inputs = self.processor(
            text=print_status_categories,
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
            predicted_print_status_idx = probs[0].argmax().item()
        
        predicted_print_status = print_status[predicted_print_status_idx]
        description_parts = []

        if predicted_print_status == "Printed":
            # Print type classification
            print_type_categories = [
                f"a photo of a {predicted_apparel} with {print_type.lower()} print pattern" 
                for print_type in print_types[:-1]  # Exclude 'Solid'
            ]
            inputs = self.processor(
                text=print_type_categories,
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
                predicted_print_type_idx = probs[0].argmax().item()
            predicted_print_type = print_types[:-1][predicted_print_type_idx]

            # Print color style classification
            color_style_categories = [
                f"a photo of a {predicted_apparel} with {style.lower()} print colors" 
                for style in print_color_styles
            ]
            inputs = self.processor(
                text=color_style_categories,
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
                predicted_color_style_idx = probs[0].argmax().item()
            predicted_color_style = print_color_styles[predicted_color_style_idx]

            # Print size classification
            size_categories = [
                f"a photo of a {predicted_apparel} with {size.lower()} print pattern" 
                for size in print_sizes
            ]
            inputs = self.processor(
                text=size_categories,
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
                predicted_size_idx = probs[0].argmax().item()
            predicted_size = print_sizes[predicted_size_idx]

            description_parts = [
                predicted_color_style.lower(),
                predicted_size.lower(),
                f"{predicted_print_type.lower()}-printed"
            ]
        else:
            # Color classification for solid items
            color_categories = [
                f"a photo of {'an' if x[0].lower() in 'aeiou' else 'a'} {x} colored clothing" 
                for x in dress_colors
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
            description_parts = [dress_colors[predicted_color_idx].lower()]

        if should_use_apparel:
            description_parts.append(predicted_apparel)
        
        return "This is a " + " ".join(description_parts)

    @contextmanager
    def tensor_management(self):
        try:
            yield
        finally:
            self._cleanup_tensors()
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def get_similar_products(self, image_path, top_k=120, page=1, items_per_page=20, search_type='combined_75'):
        with self.tensor_management():
            logger = logging.getLogger(__name__)
            try:
                logger.debug(f"Starting similarity search for {image_path} - Page {page}")
                
                # Create a cache key using only image_path and top_k
                cache_key = hashlib.md5(f"{image_path}:{top_k}:{search_type}".encode()).hexdigest()
                
                # Validate pagination parameters
                if page < 1:
                    raise ValueError("Page number must be greater than 0")
                
                # Calculate start and end indices for pagination
                start_idx = (page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                
                if start_idx >= top_k:
                    return []

                # Try to get full results from cache
                cached_results = cache.get(cache_key)
                if cached_results:
                    logger.debug(f"Cache hit for {image_path}")
                    # Extract the requested page from cached results
                    all_products = cached_results['products']
                    paginated_products = all_products[start_idx:end_idx]
                    
                    return {
                        'products': paginated_products,
                        'pagination': {
                            'current_page': page,
                            'total_pages': (top_k + items_per_page - 1) // items_per_page,
                            'total_items': top_k,
                            'items_per_page': items_per_page
                        }
                    }

                # If not in cache, process the image and get all results
                image = self.load_and_process_image(image_path)
                logger.debug("Image loaded and processed successfully")

                # Get image embedding with error handling
                try:
                    logger.debug("Generating image embeddings")
                    image_embeddings = self.get_image_embeddings(image)
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
                query_embedding = self.get_query_embedding(search_type, image_embeddings, text_description)

                # Get product results
                all_similar_products = self.search_and_retrieve_products(
                    query_embedding, 
                    search_type, 
                    top_k,
                    text_description
                )

                # Adjust pagination based on actual number of valid products
                total_valid_products = len(all_similar_products)
                total_pages = (total_valid_products + items_per_page - 1) // items_per_page

                # Ensure page number is valid
                page = min(max(1, page), total_pages) if total_pages > 0 else 1

                # Calculate start and end indices for pagination
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_valid_products)

                # Cache the full result set
                cache.set(cache_key, {'products': all_similar_products}, timeout=300)  # Cache for 5 minutes

                # Clear CUDA tensors
                if torch.cuda.is_available():
                    if hasattr(query_embedding, 'cpu'):
                        query_embedding = query_embedding.cpu()
                    torch.cuda.empty_cache()

                return {
                    'products': all_similar_products[start_idx:end_idx],
                    'pagination': {
                        'current_page': page,
                        'total_pages': total_pages,
                        'total_items': total_valid_products,
                        'items_per_page': items_per_page
                    }
                }

            except Exception as e:
                logger.error(f"Error in get_similar_products: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise

    def _monitor_memory(self, operation_name):
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logging.info(f"Memory usage after {operation_name}: {memory_mb:.2f} MB")
        if memory_mb > 1000:  # Alert if over 1GB
            logging.warning(f"High memory usage detected in {operation_name}")

    def _cleanup_tensors(self):
        for tensor in self._cached_tensors.values():
            if hasattr(tensor, 'cpu'):
                tensor.cpu()
            del tensor
        self._cached_tensors.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    def get_text_embeddings(self, text_description):
        """
        Generate text embeddings using the CLIP model.
        
        Args:
            text_description (str): Text to generate embeddings for
            
        Returns:
            numpy.ndarray: Normalized text embeddings
        """
        text_inputs = self.processor(text=[text_description], return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
        text_features = self.model.get_text_features(**text_inputs)
        text_embeddings = text_features.cpu().numpy()
        return text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        
    def get_image_embeddings(self, image):
        """
        Generate image embeddings using the CLIP model.
        
        Args:
            image: PIL image to generate embeddings for
            
        Returns:
            numpy.ndarray: Normalized image embeddings
        """
        image_inputs = self.processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            image_inputs = {k: v.cuda() for k, v in image_inputs.items()}
        image_features = self.model.get_image_features(**image_inputs)
        image_embeddings = image_features.cpu().numpy()
        return image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    def get_combined_embeddings(self, image_embeddings, text_embeddings, image_weight=0.75):
        """
        Combine image and text embeddings with specified weights.
        
        Args:
            image_embeddings: Image embeddings from get_image_embeddings
            text_embeddings: Text embeddings from get_text_embeddings
            image_weight: Weight to apply to image embeddings (1-image_weight applied to text)
            
        Returns:
            numpy.ndarray: Combined and normalized embeddings
        """
        combined = image_weight * image_embeddings + (1 - image_weight) * text_embeddings
        return combined / np.linalg.norm(combined, axis=1, keepdims=True)
    
    def get_concatenated_embeddings(self, image_embeddings, text_embeddings):
        """
        Concatenate image and text embeddings.
        
        Args:
            image_embeddings: Image embeddings from get_image_embeddings
            text_embeddings: Text embeddings from get_text_embeddings
            
        Returns:
            numpy.ndarray: Concatenated and normalized embeddings
        """
        concat = np.concatenate([image_embeddings[0], text_embeddings[0]])
        normalized = concat / np.linalg.norm(concat)
        return normalized.reshape(1, -1)

    def search_and_retrieve_products(self, query_embedding, search_type, top_k, text_description):
        """
        Search for products using the query embedding and retrieve their details.
        
        Args:
            query_embedding: The embedding to search with
            search_type: Type of search being performed
            top_k: Maximum number of results to retrieve
            text_description: Text description of the query image
            
        Returns:
            list: List of product details dictionaries
        """
        logger = logging.getLogger(__name__)
        
        # Get all results up to top_k
        distances, idx = self.indices[search_type].search(
            np.array(query_embedding, dtype="float32"),
            top_k
        )

        # Get all product details with error handling
        all_similar_products = []
        seen_products = set()  # Track seen product name+brand combinations
        
        for i in range(len(idx[0])):
            product_id = str(self.product_ids[idx[0][i]])
            try:
                product = MyntraProducts.objects.get(id=product_id)
                # Create unique key for product using name and brand
                product_key = f"{product.product_link.lower()}_{product.color.lower()}"
                
                # Skip if we've seen this product before
                if product_key in seen_products:
                    continue
                    
                seen_products.add(product_key)
                all_similar_products.append({
                    'id': product.id,
                    'product_link': product.product_link,
                    'product_name': product.name,
                    'product_price': product.price,
                    'discount_price': product.discount_price,
                    'product_image': product.image_url,
                    'product_brand': product.brand,
                    'product_marketplace': product.marketplace,
                    'similarity_score': float(distances[0][i]),
                    'description': text_description if i == 0 else None
                })
            except MyntraProducts.DoesNotExist:
                logger.warning(f"Product with ID {product_id} not found in database")
                continue
            except Exception as e:
                logger.error(f"Error processing product {product_id}: {str(e)}")
                continue
                
        return all_similar_products

    def get_query_embedding(self, search_type, image_embeddings, text_description):
        """
        Prepare the query embedding based on the specified search type.
        
        Args:
            search_type: Type of search to perform (image, text, combined_60, combined_75, concat)
            image_embeddings: Image embeddings from get_image_embeddings
            text_description: Text description of the query image
            
        Returns:
            numpy.ndarray: The prepared query embedding
        """
        if search_type == 'image':
            return image_embeddings
        elif search_type == 'text':
            return self.get_text_embeddings(text_description)
        elif search_type == 'combined_60':
            text_embeddings = self.get_text_embeddings(text_description)
            return self.get_combined_embeddings(image_embeddings, text_embeddings, 0.60)
        elif search_type == 'combined_75':
            text_embeddings = self.get_text_embeddings(text_description)
            return self.get_combined_embeddings(image_embeddings, text_embeddings, 0.75)
        elif search_type == 'concat':
            text_embeddings = self.get_text_embeddings(text_description)
            return self.get_concatenated_embeddings(image_embeddings, text_embeddings)
        else:
            raise ValueError(f"Invalid search type: {search_type}")



