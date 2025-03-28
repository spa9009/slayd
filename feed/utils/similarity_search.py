import os
# Remove the offline mode settings since they're preventing the model download
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
            logger.debug(f"Attempting to load image from: {image_path}")
            
            if image_path.startswith(('http://', 'https://')):
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    response = requests.get(image_path, headers=headers, timeout=5, stream=True)
                    print(image_path)
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

    @contextmanager
    def tensor_management(self):
        try:
            yield
        finally:
            self._cleanup_tensors()
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def get_similar_products(self, image_path, top_k=100, page=1, items_per_page=20, search_type='combined_75'):
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

                # Get all results up to top_k
                distances, idx = self.indices[search_type].search(
                    np.array(query_embedding, dtype="float32"),
                    top_k
                )

                # Get all product details with error handling
                all_similar_products = []
                for i in range(len(idx[0])):
                    product_id = str(self.product_ids[idx[0][i]])
                    try:
                        product = MyntraProducts.objects.get(id=product_id)
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


class DressClassifier:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.logger = logging.getLogger(__name__)
        
        # Basic attributes
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

        self.dress_colors = [
            "Orange", "Red", "Green", "Grey", "Pink", 
            "Blue", "Purple", "White", "Black", "Yellow", "Beige",
            "Maroon", "Burgundy", "Brown"
        ]

        # Print-related attributes
        self.print_types = [
            "Floral", "Geometric", "Polka Dots", "Leopard", "Zebra",
            "Snake", "Vertical Stripes", "Horizontal Stripes",
            "Diagonal Stripes", "Abstract", "Tie-Dye", "Paisley",
            "Checkered", "Gingham", "Solid"
        ]
        
        self.print_color_styles = [
            "Pastel", "Bright", "Neon", "Dark", "Ombre",
            "Two-Tone", "Multi-Color"
        ]
        
        self.print_sizes = [
            "Micro", "Bold", "Delicate", "Large Motif"
        ]

        self.print_status = [
            "Printed", "Solid"
        ]

    def _classify_attribute(self, image, categories, attribute_list):
        """Helper method for zero-shot classification using transformer"""
        try:
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
                probs = outputs.logits_per_image.softmax(dim=1)
                
                # Get top 3 predictions and their probabilities
                top_probs, top_indices = torch.topk(probs[0], min(3, len(attribute_list)))
                
                # Log top 3 predictions
                self.logger.debug("\nTop 3 predictions:")
                for prob, idx in zip(top_probs, top_indices):
                    self.logger.debug(f"  {attribute_list[idx]}: {prob.item():.2%}")

                if attribute_list == self.print_status:
                    printed_idx = attribute_list.index("Printed")
                    printed_prob = probs[0][printed_idx].item()
                    if printed_prob >= 0.15: 
                        return "Printed"
                    return "Solid"
                
                predicted_idx = top_indices[0].item()
                return attribute_list[predicted_idx]
                
        except Exception as e:
            self.logger.error(f"Error in _classify_attribute: {str(e)}")
            raise

    def generate_description(self, image):
        """Generate enhanced description using transformer"""
        self.logger.debug("\nStarting Classification Results:")
        self.logger.debug("="*50)

        try:
            # First determine if the dress is printed or solid
            print_status_categories = [
                f"a photo of a {status.lower()} dress" 
                for status in self.print_status
            ]
            predicted_print_status = self._classify_attribute(
                image, print_status_categories, self.print_status
            )
            self.logger.debug(f"Print Status: {predicted_print_status}")

            print_details = []
            if predicted_print_status == "Printed":
                # Print type classification
                print_type_categories = [
                    f"a photo of a dress with {print_type.lower()} print pattern" 
                    for print_type in self.print_types[:-1]  # Exclude 'Solid'
                ]
                predicted_print_type = self._classify_attribute(
                    image, print_type_categories, self.print_types[:-1]
                )
                self.logger.debug(f"Print Type: {predicted_print_type}")

                # Print color style classification
                color_style_categories = [
                    f"a photo of a dress with {style.lower()} print colors" 
                    for style in self.print_color_styles
                ]
                predicted_color_style = self._classify_attribute(
                    image, color_style_categories, self.print_color_styles
                )
                self.logger.debug(f"Print Color Style: {predicted_color_style}")

                # Print size classification
                size_categories = [
                    f"a photo of a dress with {size.lower()} print pattern" 
                    for size in self.print_sizes
                ]
                predicted_size = self._classify_attribute(
                    image, size_categories, self.print_sizes
                )
                self.logger.debug(f"Print Size: {predicted_size}")
                
                print_details = [
                    predicted_color_style.lower(),
                    predicted_size.lower(),
                    f"{predicted_print_type.lower()}-printed"
                ]

            # Basic attribute classifications
            color_categories = [
                f"a photo of {'an' if color[0].lower() in 'aeiou' else 'a'} {color} colored dress" 
                for color in self.dress_colors
            ]
            predicted_color = self._classify_attribute(image, color_categories, self.dress_colors)
            self.logger.debug(f"Color: {predicted_color}")

            length_categories = [f"a photo of a {length.lower()} dress" for length in self.dress_lengths]
            predicted_length = self._classify_attribute(image, length_categories, self.dress_lengths)
            self.logger.debug(f"Length: {predicted_length}")

            fit_categories = [f"a photo of a {fit.lower()} dress" for fit in self.dress_fits]
            predicted_fit = self._classify_attribute(image, fit_categories, self.dress_fits)
            self.logger.debug(f"Fit: {predicted_fit}")

            neckline_categories = [f"a photo of a dress with {neckline.lower()}" for neckline in self.dress_necklines]
            predicted_neckline = self._classify_attribute(image, neckline_categories, self.dress_necklines)
            self.logger.debug(f"Neckline: {predicted_neckline}")

            sleeve_categories = [f"a photo of a {sleeve.lower()} dress" for sleeve in self.dress_sleeves]
            predicted_sleeve = self._classify_attribute(image, sleeve_categories, self.dress_sleeves)
            self.logger.debug(f"Sleeves: {predicted_sleeve}")

            material_categories = [f"a photo of a {material.lower()} dress" for material in self.dress_materials]
            predicted_material = self._classify_attribute(image, material_categories, self.dress_materials)
            self.logger.debug(f"Material: {predicted_material}")

            feature_categories = [f"a photo of a dress with {feature.lower()}" for feature in self.dress_design_features]
            predicted_feature = self._classify_attribute(image, feature_categories, self.dress_design_features)
            self.logger.debug(f"Design Feature: {predicted_feature}")

            # Build description
            description_parts = [
                predicted_color.lower(),
                predicted_material.lower(),
                predicted_length.lower(),
                predicted_fit.lower(),
                predicted_neckline.lower(),
                predicted_sleeve.lower()
            ]
            
            # Add print information if present
            if print_details:
                description_parts = print_details + description_parts
            
            base_description = "This is a " + " ".join(description_parts) + " dress"
            base_description += f" with {predicted_feature.lower()}"
            
            self.logger.debug("\nFinal Description:")
            self.logger.debug("="*50)
            self.logger.debug(base_description)
            self.logger.debug("="*50)

            return base_description

        except Exception as e:
            self.logger.error(f"Error in generate_description: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
