import logging
import json
from django.conf import settings
from feed.models import DetectedObjectProducts
from utils.commn_utils import get_cdn_url
from urllib.parse import quote
from datetime import datetime
from vision.views import detect_objects
from feed.views import SimilarProductsView
from rest_framework.test import APIRequestFactory
from django.http import QueryDict

logger = logging.getLogger(__name__)

def call_vision_api(image_url):
    """
    Call Google Vision API directly using the internal function
    
    Args:
        image_url: URL of the image to process
        
    Returns:
        dict: Vision API response or None if there was an error
    """
    try:
        # Create a mock request to pass to the view function
        factory = APIRequestFactory()
        payload = {"product_url": image_url}
        request = factory.post(
            '/vision/detect-objects/',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        logger.info(f"Calling Vision API function for image: {image_url}")
        
        # Call the function directly
        response = detect_objects(request)
        
        if response.status_code == 200:
            # Parse JSON response
            vision_result = json.loads(response.content)
            logger.info(f"Vision API success: {len(vision_result.get('responses', []))} objects detected")
            return vision_result
        else:
            logger.error(f"Vision API error: {response.status_code} - {response.content}")
            return None
            
    except Exception as e:
        logger.exception(f"Error calling Vision API function: {str(e)}")
        return None

def get_similar_products_for_whole_image(image_url, vision_result=None):
    """
    Get similar products for an entire image.
    Used as a fallback when no objects are detected.
    
    Args:
        image_url: URL of the image to find similar products for
        vision_result: Vision API result to store
        
    Returns:
        dict: Contains success status and product IDs
    """
    try:
        # Check if we already have a whole-image result for this URL
        existing_result = DetectedObjectProducts.objects.filter(
            image_url=image_url,
            is_whole_image=True
        ).first()
        
        if existing_result:
            logger.info(f"Found cached whole-image results for {image_url}")
            return {
                "success": True,
                "product_ids": existing_result.similar_products,
                "existing": True
            }
        
        # Call the similar-products view directly
        factory = APIRequestFactory()
        query_params = QueryDict('', mutable=True)
        query_params.update({
            "image_url": image_url,
            "search_type": "combined_75",
            "page": 1,
            "items_per_page": 30
        })
        
        # Create a GET request with query parameters
        request = factory.get('/feed/similar-products/', query_params)
        
        logger.info(f"Calling similar-products function for whole image: {image_url}")
        
        # Create view instance and call it directly
        view = SimilarProductsView.as_view()
        response = view(request)
        
        if response.status_code == 200:
            data = json.loads(response.rendered_content)
            
            # Extract product IDs from the response
            product_ids = []
            if "products" in data and isinstance(data["products"], list):
                product_ids = [product.get("id") for product in data["products"] if product.get("id")]
                logger.info(f"Found {len(product_ids)} similar products for whole image")
            
            # Store results in DetectedObjectProducts model as a whole-image entry
            if product_ids:
                # Create whole-image entry
                DetectedObjectProducts.objects.create(
                    image_url=image_url,
                    label="whole_image",
                    x=0.0,
                    y=0.0,
                    width=1.0,
                    height=1.0,
                    confidence=1.0,
                    similar_products=product_ids,
                    is_whole_image=True,
                    vision_result=vision_result
                )
                logger.info(f"Stored whole-image results in database for {image_url}")
            
            return {
                "success": True,
                "product_ids": product_ids,
                "existing": False
            }
            
        else:
            logger.error(f"Similar products function error: {response.status_code}")
            return {
                "success": False,
                "error": f"Function Error: {response.status_code}",
                "existing": False
            }
            
    except Exception as e:
        logger.exception(f"Error getting similar products for whole image: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "existing": False
        }

def get_similar_products_for_object(image_url, x, y, width, height, label, confidence):
    """
    Get similar products for a specific detected object in an image
    
    Args:
        image_url: URL of the image containing the object
        x: X coordinate of object bounding box (top-left)
        y: Y coordinate of object bounding box (top-left)
        width: Width of object bounding box
        height: Height of object bounding box
        label: Object label/class
        confidence: Detection confidence
        
    Returns:
        dict: Contains:
            - success (bool): Whether the operation succeeded
            - product_ids (list): List of product IDs if successful
            - error (str): Error message if not successful
    """
    try:
        # First check if we already have results for this object
        existing_object = DetectedObjectProducts.objects.filter(
            image_url=image_url,
            label=label,
            x=x,
            y=y,
            width=width,
            height=height,
            is_whole_image=False
        ).first()
        
        if existing_object:
            logger.info(f"Found existing similar products for object {label} in {image_url}")
            return {
                "success": True,
                "product_ids": existing_object.similar_products,
                "existing": True,
                "object_id": existing_object.id
            }
        
        # Call the similar-products view with the cropped region
        factory = APIRequestFactory()
        query_params = QueryDict('', mutable=True)
        query_params.update({
            "image_url": image_url,  # Original image URL
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "search_type": "combined_75",
            "page": 1,
            "items_per_page": 20
        })
        
        # Create a GET request with query parameters
        request = factory.get('/feed/similar-products/', query_params)
        
        logger.info(f"Finding similar products for {label} in {image_url}")
        
        # Create view instance and call it directly
        view = SimilarProductsView.as_view()
        response = view(request)
        
        if response.status_code == 200:
            data = json.loads(response.rendered_content)
            
            # Extract product IDs from the response
            product_ids = []
            if "products" in data and isinstance(data["products"], list):
                product_ids = [product.get("id") for product in data["products"] if product.get("id")]
                logger.info(f"Found {len(product_ids)} similar products for {label}")
            
            # Store results in DetectedObjectProducts model
            obj = None
            if product_ids:
                obj = DetectedObjectProducts.objects.create(
                    image_url=image_url,
                    label=label,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=confidence,
                    similar_products=product_ids,
                    is_whole_image=False
                )
                logger.info(f"Stored similar products for {label} in database")
            
            return {
                "success": True,
                "product_ids": product_ids,
                "existing": False,
                "object_id": obj.id if obj else None
            }
            
        else:
            logger.error(f"Similar products function error: {response.status_code}")
            return {
                "success": False,
                "error": f"Function Error: {response.status_code}"
            }
            
    except Exception as e:
        logger.exception(f"Error getting similar products for object: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def process_image_for_similar_products(image_url):
    """
    Process an image through Vision API and get similar products
    
    This version ONLY gets similar products for the whole image, used as a fallback.
    
    Args:
        image_url: URL of the image to process
        
    Returns:
        dict: Result information
    """
    try:
        # Call Vision API
        vision_result = call_vision_api(image_url)
        
        if not vision_result:
            return {
                "success": False,
                "error": "Failed to call Vision API or no results returned"
            }
        
        # Get similar products for the whole image
        products_result = get_similar_products_for_whole_image(image_url, vision_result)
        
        if not products_result["success"]:
            return {
                "success": False,
                "error": products_result.get("error", "No similar products found")
            }
        
        return {
            "success": True,
            "product_ids": products_result["product_ids"],
            "vision_result": vision_result,
            "existing": products_result.get("existing", False)
        }
        
    except Exception as e:
        logger.exception(f"Error processing image for similar products: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def process_objects_in_image(image_url):
    """
    Process all detected objects in an image and get similar products for each
    
    Workflow:
    1. Call Vision API to detect objects
    2. For each detected object, get similar products
    3. If no objects are detected, get similar products for the whole image
    
    Args:
        image_url: URL of the image to process
        
    Returns:
        dict: Result information including:
            - success (bool): Whether the operation succeeded
            - object_results (list): List of objects with their product IDs
            - vision_result (dict): Vision API response
            - error (str): Error message if not successful
    """
    try:
        # Call Vision API
        vision_result = call_vision_api(image_url)
        
        if not vision_result:
            return {
                "success": False,
                "error": "Failed to call Vision API or no results returned"
            }
        
        # Process each detected object
        object_results = []
        has_detected_objects = False
        
        # Check if we have object detections in the vision result
        if vision_result and "responses" in vision_result:
            for response in vision_result["responses"]:
                if "items" in response and isinstance(response["items"], list) and len(response["items"]) > 0:
                    has_detected_objects = True
                    # Process each detected item/object
                    for item in response["items"]:
                        # Extract object details
                        label = item.get("label", "unknown")
                        confidence = item.get("confidence", 0.0)
                        x = item.get("x", 0.0)
                        y = item.get("y", 0.0)
                        box_width = item.get("boxWidth", 0.0)
                        box_height = item.get("boxHeight", 0.0)
                            
                        # Get similar products for this object
                        object_result = get_similar_products_for_object(
                            image_url=image_url,
                            x=x,
                            y=y,
                            width=box_width,
                            height=box_height,
                            label=label,
                            confidence=confidence
                        )
                        
                        # Add object result to our list
                        object_results.append({
                            "id": object_result.get("object_id"),
                            "label": label,
                            "confidence": confidence,
                            "x": x,
                            "y": y,
                            "width": box_width,
                            "height": box_height,
                            "success": object_result["success"],
                            "product_ids": object_result.get("product_ids", [])
                        })
        
        # If no objects were detected, get similar products for the whole image
        whole_image_products = []
        if not has_detected_objects:
            logger.info("No objects detected, getting similar products for the whole image")
            whole_image_result = get_similar_products_for_whole_image(image_url, vision_result)
            
            if whole_image_result["success"]:
                whole_image_products = whole_image_result["product_ids"]
        
        return {
            "success": True,
            "object_results": object_results,
            "whole_image_products": whole_image_products,
            "has_detected_objects": has_detected_objects,
            "vision_result": vision_result
        }
        
    except Exception as e:
        logger.exception(f"Error processing objects in image: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def get_product_card_data(image_url, sender_id=None):
    """
    Get data needed for a product card, including similar products
    
    Args:
        image_url: URL of the image to find similar products for
        sender_id: Optional Instagram sender ID for tracking
        
    Returns:
        dict: Product card data including:
            - success (bool): Whether the operation succeeded
            - card_data (dict): Data for creating a product card if successful
            - error (str): Error message if not successful
    """
    try:
        # Process image and get similar products for all objects
        result = process_objects_in_image(image_url)
        
        if not result["success"]:
            logger.error(f"Failed to get similar products: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error", "Failed to process image")
            }
        
        # Collect all product IDs
        all_product_ids = []
        
        # If we have detected objects with products, use those
        if result.get("has_detected_objects", False):
            for obj in result.get("object_results", []):
                if obj.get("success") and obj.get("product_ids"):
                    all_product_ids.extend(obj.get("product_ids", []))
        # Otherwise, use whole-image products
        else:
            all_product_ids = result.get("whole_image_products", [])

        # Create the URL for the button
        button_url = f"https://slayd.in/similar-product/?image_url={quote(image_url)}"
        
        return {
            "success": True,
            "card_data": {
                "image_url": image_url,
                "title": "Check out these similar products! 🛍️",
                "subtitle": "We found great matches for items in your image",
                "button_url": button_url
            },
            "has_detected_objects": result.get("has_detected_objects", False),
            "object_results": result.get("object_results", []),
            "whole_image_products": result.get("whole_image_products", []),
            "vision_result": result.get("vision_result")
        }
        
    except Exception as e:
        logger.exception(f"Error creating product card data: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        } 