import os
import numpy as np
import faiss
from PIL import Image
import requests
from io import BytesIO
from feed.models import MyntraProducts
from feed.model_loader import get_model_instance
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
                self.fclip = model_instance.fclip
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

    def _cleanup_tensors(self):
        for tensor in self._cached_tensors.values():
            if hasattr(tensor, 'cpu'):
                tensor.cpu()
            del tensor
        self._cached_tensors.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _crop_image(self, image, x, y, width, height):
        """
        Crop an image based on normalized coordinates (0-1).
        
        Args:
            image: PIL Image object to crop
            x, y: Top-left corner coordinates (normalized 0-1)
            width, height: Width and height of crop (normalized 0-1)
            
        Returns:
            Cropped PIL Image
        """
        logger = logging.getLogger(__name__)
        try:
            logger.debug(f"Cropping image with coordinates: x={x}, y={y}, width={width}, height={height}")
            
            # Convert normalized coordinates (0-1) to pixel values
            img_width, img_height = image.size
            x_pixel = int(float(x) * img_width)
            y_pixel = int(float(y) * img_height)
            width_pixel = int(float(width) * img_width)
            height_pixel = int(float(height) * img_height)
            
            # Ensure valid crop area
            x_pixel = max(0, x_pixel)
            y_pixel = max(0, y_pixel)
            width_pixel = min(width_pixel, img_width - x_pixel)
            height_pixel = min(height_pixel, img_height - y_pixel)
            
            # Crop the image
            cropped_image = image.crop((x_pixel, y_pixel, x_pixel + width_pixel, y_pixel + height_pixel))
            logger.debug(f"Image cropped successfully to size {cropped_image.size}")
            
            return cropped_image
        except Exception as e:
            logger.error(f"Error cropping image: {str(e)}")
            # Return original image if cropping fails
            return image

    def _preprocess_image(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image.resize((224, 224))

    def _fetch_remote_image(self, image_url):
        logger = logging.getLogger(__name__)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(image_url, headers=headers, timeout=5, stream=True)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

    def load_and_process_image(self, image_path, x=None, y=None, width=None, height=None):
        logger = logging.getLogger(__name__)
        try:
            # Load image from URL or local path
            if image_path.startswith(('http://', 'https://')):
                image = self._fetch_remote_image(image_path)
            else:
                image = Image.open(image_path)
            
            # Apply cropping if coordinates are provided
            if all(param is not None for param in [x, y, width, height]):
                image = self._crop_image(image, x, y, width, height)
            
            # Preprocess the image
            return self._preprocess_image(image)
            
        except Exception as e:
            logger.error(f"Error in load_and_process_image: {str(e)}")
            raise

    def _get_image_embeddings(self, image):
        logger = logging.getLogger(__name__)
        try:
            logger.debug("Generating image embeddings")
            image_embeddings = self.fclip.encode_images(images=[image], batch_size=1)
            image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
            logger.debug("Image embeddings generated successfully")
            return image_embeddings
        except Exception as e:
            logger.error(f"Failed to generate image embeddings: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")

    def _generate_query_embedding(self, image_embeddings, text_description, search_type):

        logger = logging.getLogger(__name__)
        try:
            # Prepare query embedding based on search type
            if search_type == 'image':
                return image_embeddings
            elif search_type == 'text':
                text_embedding = self.fclip.encode_text([text_description], batch_size=1)
                return text_embedding / np.linalg.norm(text_embedding, ord=2, axis=-1, keepdims=True)
            elif search_type == 'combined_60':
                text_embedding = self.fclip.encode_text([text_description], batch_size=1)
                combined = 0.60 * image_embeddings + 0.40 * text_embedding
                return combined / np.linalg.norm(combined, ord=2, axis=-1, keepdims=True)
            elif search_type == 'combined_75':
                text_embedding = self.fclip.encode_text([text_description], batch_size=1)
                combined = 0.75 * image_embeddings + 0.25 * text_embedding
                return combined / np.linalg.norm(combined, ord=2, axis=-1, keepdims=True)
            elif search_type == 'concat':
                text_embedding = self.fclip.encode_text([text_description], batch_size=1)
                concatenated = np.concatenate([image_embeddings[0], text_embedding[0]])
                concatenated = concatenated / np.linalg.norm(concatenated)
                return concatenated.reshape(1, -1)
            else:
                raise ValueError(f"Invalid search type: {search_type}")
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def _get_product_details(self, product_id, distances=None, idx=None):

        logger = logging.getLogger(__name__)
        try:
            product = MyntraProducts.objects.get(id=product_id)
            similarity_score = float(distances[0][idx]) if distances is not None and idx is not None else None
            
            return {
                'id': product.id,
                'product_link': product.product_link,
                'product_name': product.name,
                'product_price': product.price,
                'discount_price': product.discount_price,
                'product_image': product.image_url,
                'product_brand': product.brand,
                'product_marketplace': product.marketplace,
                'similarity_score': similarity_score
            }
        except MyntraProducts.DoesNotExist:
            logger.warning(f"Product with ID {product_id} not found in database")
            return None
        except Exception as e:
            logger.error(f"Error processing product {product_id}: {str(e)}")
            return None

    def get_text_description(self, image):
        """Generate detailed text description using CLIP zero-shot classification"""
        apparel_types = [
            "dress", "kurta", "top", "shirt", "pants", "skirt", "suit", "pants", "jeans", "shorts",
            "saree", "t-shirt", "jacket", "lehenga", "joggers", "skorts", "sweatshirt", "hoodie",
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
        if predicted_apparel == "dress" :
            dress_classifier = DressClassifier(self.model, self.processor, predicted_apparel)
            detailed_description = dress_classifier.generate_description(image)
            return detailed_description
        elif predicted_apparel == "top" or predicted_apparel == "bralette":
            top_classifier = TopsClassifier(self.model, self.processor, predicted_apparel)
            detailed_description = top_classifier.generate_description(image)
            return detailed_description
        else:
            dress_colors = [
                "Orange", "Red", "Green", "Grey", "Pink", 
                "Blue", "Purple", "White", "Black", "Yellow", "Beige",
                "Maroon", "Burgundy", "Brown"
            ]
            color_categories = [
                f"a photo of {'an' if x[0].lower() in 'aeiou' else 'a'} {x} colored clothing" 
                for x in dress_colors  # Use dress_classifier's colors
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
            
            predicted_color = dress_colors[predicted_color_idx]
            return f"This is a {predicted_color.lower()} {predicted_apparel}"

    @contextmanager
    def tensor_management(self):
        try:
            yield
        finally:
            self._cleanup_tensors()
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def get_similar_products(self, image_path, top_k=120, page=1, items_per_page=20, search_type='combined_75', x=None, y=None, width=None, height=None):
        with self.tensor_management():
            logger = logging.getLogger(__name__)
            try:
                logger.debug(f"Starting similarity search for {image_path} - Page {page}")
                
                # Create a cache key using image_path, crop parameters, and search_type
                crop_params = f":{x}:{y}:{width}:{height}" if all(param is not None for param in [x, y, width, height]) else ""
                cache_key = hashlib.md5(f"{image_path}{crop_params}:{top_k}:{search_type}".encode()).hexdigest()
                
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
                    logger.debug(f"Cache hit for {image_path}{crop_params}")
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
                image = self.load_and_process_image(image_path, x, y, width, height)
                logger.debug("Image loaded and processed successfully")

                # Get image embedding
                image_embeddings = self._get_image_embeddings(image)

                # Get text description with error handling
                try:
                    logger.debug("Generating text description")
                    text_description = self.get_text_description(image)
                    logger.debug(f"Generated text description: {text_description}")
                except Exception as e:
                    logger.error(f"Failed to generate text description: {str(e)}")
                    raise RuntimeError(f"Text description generation failed: {str(e)}")

                # Generate query embedding
                query_embedding = self._generate_query_embedding(image_embeddings, text_description, search_type)

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
                    product_details = self._get_product_details(product_id, distances, i)
                    
                    if product_details:
                        # Add description only to the first product
                        if i == 0:
                            product_details['description'] = text_description
                            
                        # Create unique key for product using product link and color
                        product_key = f"{product_details['product_link'].lower()}_{product_details.get('color', '').lower()}"
                        
                        # Skip if we've seen this product before
                        if product_key in seen_products:
                            continue
                        
                        seen_products.add(product_key)
                        all_similar_products.append(product_details)

                # Adjust pagination based on actual number of valid products
                total_valid_products = len(all_similar_products)
                total_pages = (total_valid_products + items_per_page - 1) // items_per_page if total_valid_products > 0 else 1

                # Ensure page number is valid
                page = min(max(1, page), total_pages) if total_pages > 0 else 1

                # Calculate start and end indices for pagination
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_valid_products)

                # Cache the full result set
                cache.set(cache_key, {'products': all_similar_products}, timeout=300)  # Cache for 5 minutes

                # Clear CUDA tensors
                if torch.cuda.is_available() and hasattr(query_embedding, 'values'):
                    for v in query_embedding.values():
                        if hasattr(v, 'cpu'):
                            v.cpu()
                    del query_embedding
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



