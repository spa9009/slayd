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
from feed.utils.classifier.jeans_classifier import JeansClassifier
from feed.utils.classifier.skirt_classifier import SkirtClassifier
from feed.utils.classifier.pant_classifier import PantClassifier

# Configure logger
logger = logging.getLogger(__name__)

class SingletonMeta(type):
    """Metaclass to implement the Singleton pattern."""
    _instances = {}
    _lock = Lock()
    
    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ImageProcessor:
    """Component responsible for loading and processing images."""
    
    @staticmethod
    def crop_image(image, x, y, width, height):
        """
        Crop an image based on normalized coordinates (0-1).
        
        Args:
            image: PIL Image object to crop
            x, y: Top-left corner coordinates (normalized 0-1)
            width, height: Width and height of crop (normalized 0-1)
            
        Returns:
            Cropped PIL Image
        """
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

    @staticmethod
    def preprocess_image(image):
        """Convert image to RGB and resize to standard dimensions."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image.resize((224, 224))

    @staticmethod
    def fetch_remote_image(image_url):
        """Fetch an image from a remote URL."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(image_url, headers=headers, timeout=5, stream=True)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

    @classmethod
    def load_and_process(cls, image_path, x=None, y=None, width=None, height=None):
        """Load, crop (if needed) and preprocess an image."""
        try:
            # Load image from URL or local path
            if image_path.startswith(('http://', 'https://')):
                image = cls.fetch_remote_image(image_path)
            else:
                image = Image.open(image_path)
            
            # Apply cropping if coordinates are provided
            if all(param is not None for param in [x, y, width, height]):
                image = cls.crop_image(image, x, y, width, height)
            
            # Preprocess the image
            return cls.preprocess_image(image)
            
        except Exception as e:
            logger.error(f"Error in load_and_process_image: {str(e)}")
            raise


class TensorManager:
    """Component for managing tensors and memory."""
    
    def __init__(self):
        self._cached_tensors = {}
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def cleanup(self):
        """Clean up tensors and free memory."""
        for tensor in self._cached_tensors.values():
            if hasattr(tensor, 'cpu'):
                tensor.cpu()
            del tensor
        self._cached_tensors.clear()
        
        # Free CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()
    
    @contextmanager
    def managed_context(self):
        """Context manager for tensor operations with automatic cleanup."""
        try:
            yield
        finally:
            self.cleanup()
    
    def monitor_memory(self, operation_name):
        """Monitor memory usage and log warnings if high."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Memory usage after {operation_name}: {memory_mb:.2f} MB")
        if memory_mb > 1000:  # Alert if over 1GB
            logger.warning(f"High memory usage detected in {operation_name}")


class ClassifierFactory:
    """Factory for creating and managing apparel classifiers."""
    
    @staticmethod
    def get_classifier(apparel_type, model, processor):
        """Return the appropriate classifier for a given apparel type."""
        if apparel_type == "dress":
            return DressClassifier(model, processor, apparel_type)
        elif apparel_type in ["top", "bralette"]:
            return TopsClassifier(model, processor, apparel_type)
        elif apparel_type == "jeans":
            return JeansClassifier(model, processor, apparel_type)
        elif apparel_type in ["pants", "shorts"]:
            return PantClassifier(model, processor, "pants")
        elif apparel_type in ["skirt", "mini skirt", "denim skirt"]:
            return SkirtClassifier(model, processor, "skirt")
        return None


class QueryEmbeddingGenerator:
    """Component for generating query embeddings from images and text."""
    
    def __init__(self, clip_model, clip_processor):
        self.model = clip_model
        self.processor = clip_processor
    
    def get_image_embeddings(self, image):
        """Generate embeddings from an image."""
        try:
            image_inputs = self.processor(images=image, return_tensors="pt")
            if torch.cuda.is_available():
                image_inputs = {k: v.cuda() for k, v in image_inputs.items()}
            image_features = self.model.get_image_features(**image_inputs)
            image_embedding = image_features.cpu().numpy()
            image_embedding = image_embedding / np.linalg.norm(image_embedding, ord=2, axis=-1, keepdims=True)
            logger.debug("Image embeddings generated successfully")
            return image_embedding
        except Exception as e:
            logger.error(f"Failed to generate image embeddings: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
    
    def get_text_embeddings(self, text):
        """Generate embeddings from a text description."""
        try:
            # Get text embedding
            text_inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
            text_features = self.model.get_text_features(**text_inputs)
            text_embedding = text_features.cpu().numpy()
            return text_embedding / np.linalg.norm(text_embedding, ord=2, axis=-1, keepdims=True)
        except Exception as e:
            logger.error(f"Failed to generate text embeddings: {str(e)}")
            raise RuntimeError(f"Text embedding generation failed: {str(e)}")
    
    def generate_query_embedding(self, image_embeddings, text_description, search_type):
        """Generate combined embeddings based on search type."""
        try:
            if search_type == 'image':
                return image_embeddings
            elif search_type == 'text':
                return self.get_text_embeddings(text_description)
            elif search_type == 'combined_60':
                text_embedding = self.get_text_embeddings(text_description)
                combined = 0.60 * image_embeddings + 0.40 * text_embedding
                return combined / np.linalg.norm(combined, ord=2, axis=-1, keepdims=True)
            elif search_type == 'combined_75':
                text_embedding = self.get_text_embeddings(text_description)
                combined = 0.75 * image_embeddings + 0.25 * text_embedding
                return combined / np.linalg.norm(combined, ord=2, axis=-1, keepdims=True)
            elif search_type == 'concat':
                text_embedding = self.get_text_embeddings(text_description)
                concatenated = np.concatenate([image_embeddings[0], text_embedding[0]])
                concatenated = concatenated / np.linalg.norm(concatenated)
                return concatenated.reshape(1, -1)
            else:
                raise ValueError(f"Invalid search type: {search_type}")
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise


class ProductDetailsManager:
    """Component for retrieving and formatting product details."""
    
    @staticmethod
    def get_product_details(product_id, distances=None, idx=None):
        """Get product details from database and format them."""
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


class TextDescriptionGenerator:
    """Component for generating descriptive text for images."""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        
    def _get_apparel_type(self, image):
        """Identify the type of apparel in the image."""
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
        
        return apparel_types[predicted_apparel_idx]
    
    def classify_print_status(self, image, predicted_apparel):
        """Determine if the item is printed or solid."""
        print_status = ["Printed", "Solid"]
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
        
        return print_status[predicted_print_status_idx]
    
    def classify_with_categories(self, image, categories):
        """General-purpose classifier using provided categories."""
        inputs = self.processor(
            text=categories,
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
            predicted_idx = probs[0].argmax().item()
        
        return predicted_idx
    
    def get_basic_description(self, image, predicted_apparel, should_use_apparel=True):
        """
        Generate a basic description for apparel items.
        """
        # Print-related attributes
        print_types = [
            "Floral", "Geometric", "Abstract", "Animal", "Striped", 
            "Polka Dot", "Paisley", "Solid"
        ]
        print_color_styles = ["Monochrome", "Multicolor"]
        print_sizes = ["Small", "Medium", "Large"]
        
        # Colors for solid items
        dress_colors = [
            "Orange", "Red", "Green", "Grey", "Pink", 
            "Blue", "Purple", "White", "Black", "Yellow", "Beige",
            "Maroon", "Burgundy", "Brown"
        ]

        # First determine if the item is printed or solid
        predicted_print_status = self.classify_print_status(image, predicted_apparel)
        description_parts = []

        if predicted_print_status == "Printed":
            # Print type classification
            print_type_categories = [
                f"a photo of a {predicted_apparel} with {print_type.lower()} print pattern" 
                for print_type in print_types[:-1]  # Exclude 'Solid'
            ]
            predicted_print_type_idx = self.classify_with_categories(image, print_type_categories)
            predicted_print_type = print_types[:-1][predicted_print_type_idx]

            # Print color style classification
            color_style_categories = [
                f"a photo of a {predicted_apparel} with {style.lower()} print colors" 
                for style in print_color_styles
            ]
            predicted_color_style_idx = self.classify_with_categories(image, color_style_categories)
            predicted_color_style = print_color_styles[predicted_color_style_idx]

            # Print size classification
            size_categories = [
                f"a photo of a {predicted_apparel} with {size.lower()} print pattern" 
                for size in print_sizes
            ]
            predicted_size_idx = self.classify_with_categories(image, size_categories)
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
            
            predicted_color_idx = self.classify_with_categories(image, color_categories)
            description_parts = [dress_colors[predicted_color_idx].lower()]

        if should_use_apparel:
            description_parts.append(predicted_apparel)
        
        return " ".join(description_parts)
    
    def generate_description(self, image):
        """Generate a detailed text description for an image."""
        predicted_apparel = self._get_apparel_type(image)
        
        # Use specialized classifiers for specific apparel types
        classifier = ClassifierFactory.get_classifier(predicted_apparel, self.model, self.processor)
        
        if classifier:
            return classifier.generate_description(image)
        elif predicted_apparel == "Indian Ethnic":
            return self.get_basic_description(image, predicted_apparel, False)
        else:
            return self.get_basic_description(image, predicted_apparel, True)


class PaginationHelper:
    """Helper class for pagination-related operations."""
    
    @staticmethod
    def validate_and_calculate_indices(page, items_per_page, total_items):
        """Validate page number and calculate start/end indices."""
        if page < 1:
            raise ValueError("Page number must be greater than 0")
        
        # Calculate indices for pagination
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        # Make sure start index is within bounds
        if start_idx >= total_items:
            start_idx = 0
            end_idx = min(items_per_page, total_items)
            page = 1
        
        return start_idx, end_idx, page
    
    @staticmethod
    def generate_pagination_info(page, items_per_page, total_items):
        """Generate pagination information dictionary."""
        total_pages = (total_items + items_per_page - 1) // items_per_page if total_items > 0 else 1
        page = min(max(1, page), total_pages)
        
        return {
            'current_page': page,
            'total_pages': total_pages,
            'total_items': total_items,
            'items_per_page': items_per_page
        }


class SimilaritySearcher(metaclass=SingletonMeta):
    """Main class for finding similar products using image/text embeddings."""
    
    def __init__(self):
        self._is_initialized = False
        self.tensor_manager = TensorManager()
        self._initialize()
    
    def _initialize(self):
        """Initialize models, indices, and other required components."""
        if self._is_initialized:
            return
            
        try:
            logger.debug("Starting initialization...")
            if settings.DEBUG:
                base_path = os.path.join(settings.BASE_DIR, 'indices_ivf')
            else:
                base_path = getattr(settings, 'FAISS_INDICES_PATH', '/opt/fashion_recommendation/indices')
            
            logger.debug(f"Using base path: {base_path}")
            
            # Initialize models
            logger.debug("Initializing models...")
            try:
                model_instance = get_model_instance()
                self.model = model_instance.clip_model
                self.processor = model_instance.clip_processor
                self.fclip = model_instance.fclip  # Ensure this exists in your model_loader
                logger.debug("Models initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize models: {str(e)}")
                raise
            
            # Initialize components
            self.image_processor = ImageProcessor()
            self.embedding_generator = QueryEmbeddingGenerator(self.model, self.processor)
            self.description_generator = TextDescriptionGenerator(self.model, self.processor)
            self.product_manager = ProductDetailsManager()
            self.pagination_helper = PaginationHelper()
            
            # Load all FAISS indices
            logger.debug("Loading FAISS indices...")
            try:
                self.indices = {
                    'combined_75': self._load_index(os.path.join(base_path, 'combined_faiss_index_75_ivf50.bin')),
                }
                # Set nprobe parameter for IVF index
                self.indices['combined_75'].nprobe = 50
                logger.debug("Loaded all indices successfully")
                
                self.product_ids = self._load_product_ids(os.path.join(base_path, 'product_ids.npy'))
                logger.debug("Loaded product IDs")
            except Exception as e:
                logger.error(f"Failed to load FAISS indices: {str(e)}")
                raise
            
            self._is_initialized = True
            logger.debug("Initialization completed successfully")

        except Exception as e:
            logger.error(f"Initialization error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize SimilaritySearcher: {str(e)}")

    def _load_index(self, path):
        """Load a FAISS index from a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index not found at {path}")
        return faiss.read_index(path)

    def _load_product_ids(self, path):
        """Load product IDs from a NumPy file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Product IDs file not found at {path}")
        return np.load(path)
    
    def _create_cache_key(self, image_path, x, y, width, height, top_k, search_type):
        """Create a cache key for the search results."""
        crop_params = f":{x}:{y}:{width}:{height}" if all(param is not None for param in [x, y, width, height]) else ""
        return hashlib.md5(f"{image_path}{crop_params}:{top_k}:{search_type}".encode()).hexdigest()
    
    def _get_products_from_search_results(self, distances, idx, text_description=None):
        """Convert search results to product details."""
        all_similar_products = []
        seen_products = set()  # Track seen product combinations
        
        for i in range(len(idx[0])):
            product_id = str(self.product_ids[idx[0][i]])
            try:
                product = MyntraProducts.objects.get(id=product_id)
                
                # Create unique key for product using link and color with null checks
                product_link = product.product_link.lower() if product.product_link else ""
                product_color = product.color.lower() if product.color else ""
                product_key = f"{product_link}_{product_color}"
                
                # Skip if we've seen this product before
                if product_key in seen_products and product_key:  # Only skip if key is not empty
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
                logger.error(f"Error1 processing product {product_id}: {str(e)}")
                continue
                
        return all_similar_products
    
    def get_similar_products(self, image_path, top_k=120, page=1, items_per_page=20, 
                            search_type='combined_75', x=None, y=None, width=None, height=None):
        """
        Find products similar to the input image.
        
        Args:
            image_path: Path or URL to the query image
            top_k: Maximum number of results to return
            page: Page number for pagination
            items_per_page: Number of items per page
            search_type: Type of search (image, text, combined_75, etc.)
            x, y, width, height: Optional cropping parameters (normalized 0-1)
            
        Returns:
            Dictionary with paginated products and pagination info
        """
        with self.tensor_manager.managed_context():
            try:
                logger.debug(f"Starting similarity search for {image_path} - Page {page}")
                
                # Create cache key and check cache
                cache_key = self._create_cache_key(image_path, x, y, width, height, top_k, search_type)
                cached_results = cache.get(cache_key)
                
                if cached_results:
                    logger.debug(f"Cache hit for {image_path}")
                    all_products = cached_results['products']
                    total_products = len(all_products)
                    
                    # Calculate pagination indices
                    start_idx, end_idx, page = self.pagination_helper.validate_and_calculate_indices(
                        page, items_per_page, total_products
                    )
                    
                    return {
                        'products': all_products[start_idx:end_idx],
                        'pagination': self.pagination_helper.generate_pagination_info(
                            page, items_per_page, total_products
                        )
                    }

                # Process the image
                image = ImageProcessor.load_and_process(image_path, x, y, width, height)
                logger.debug("Image loaded and processed successfully")
                
                # Get image embeddings
                image_embeddings = self.embedding_generator.get_image_embeddings(image)
                
                # Generate text description
                try:
                    logger.debug("Generating text description")
                    text_description = self.description_generator.generate_description(image)
                    logger.debug(f"Generated text description: {text_description}")
                except Exception as e:
                    logger.error(f"Failed to generate text description: {str(e)}")
                    raise RuntimeError(f"Text description generation failed: {str(e)}")
                
                # Generate query embedding
                query_embedding = self.embedding_generator.generate_query_embedding(
                    image_embeddings, text_description, search_type
                )
                
                # Perform search with FAISS
                distances, idx = self.indices[search_type].search(
                    np.array(query_embedding, dtype="float32"),
                    top_k
                )
                
                # Get product details
                all_similar_products = self._get_products_from_search_results(
                    distances, idx, text_description
                )
                
                # Calculate pagination
                total_valid_products = len(all_similar_products)
                
                # Calculate pagination indices
                start_idx, end_idx, page = self.pagination_helper.validate_and_calculate_indices(
                    page, items_per_page, total_valid_products
                )
                
                # Cache the results
                cache.set(cache_key, {'products': all_similar_products}, timeout=300)  # 5 minutes
                
                return {
                    'products': all_similar_products[start_idx:end_idx],
                    'pagination': self.pagination_helper.generate_pagination_info(
                        page, items_per_page, total_valid_products
                    )
                }

            except Exception as e:
                logger.error(f"Error in get_similar_products: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise



