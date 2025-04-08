from django.shortcuts import render
import json
import base64
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import logging
from io import BytesIO

logger = logging.getLogger(__name__)

# Define clothing categories to filter results
CLOTHING_CATEGORIES = [
  'shirt', 'dress', 'pants', 'skirt', 'jacket', 'coat', 'sweater',
  'blouse', 'top', 'jeans', 'shorts', 'suit', 'blazer', 'hoodie',
  't-shirt', 'tshirt', 'sweatshirt', 'cardigan', 'vest', 'jumpsuit', 'outerwear'
]

def filter_clothing_items(response_data):
    if 'responses' not in response_data:
        return response_data
        
    for response in response_data['responses']:
        if 'localizedObjectAnnotations' not in response:
            continue
            
        filtered_annotations = []
        
        for annotation in response['localizedObjectAnnotations']:
            name_lower = annotation['name'].lower()
            
            # Check if item is clothing
            is_clothing = name_lower in CLOTHING_CATEGORIES or any(
                category in name_lower for category in CLOTHING_CATEGORIES
            )
            
            if is_clothing:
                filtered_annotations.append(annotation)
                logger.info(f"Keeping clothing item: {annotation['name']}")
            else:
                logger.info(f"Filtering out non-clothing item: {annotation['name']}")
        
        # Replace with filtered results
        response['localizedObjectAnnotations'] = filtered_annotations
    
    return response_data

def transform_response(response_data):
    """
    Transform the Google Vision API response to include more usable fields.
    
    Adds:
    - label: The item name
    - confidence: The detection confidence score
    - x: Top-left corner x-coordinate
    - y: Top-left corner y-coordinate
    - boxWidth: Width of the bounding box
    - boxHeight: Height of the bounding box
    """
    if 'responses' not in response_data:
        return response_data
        
    for response in response_data['responses']:
        if 'localizedObjectAnnotations' not in response:
            continue
            
        transformed_items = []
        
        for annotation in response['localizedObjectAnnotations']:
            if 'boundingPoly' not in annotation or 'normalizedVertices' not in annotation['boundingPoly']:
                continue
                
            vertices = annotation['boundingPoly']['normalizedVertices']
            if len(vertices) < 4:
                continue
                
            # Get top-left and bottom-right vertices
            top_left = vertices[0]  # Top-left corner
            bottom_right = vertices[2]  # Bottom-right corner
            
            # Calculate dimensions
            box_width = bottom_right['x'] - top_left['x']
            box_height = bottom_right['y'] - top_left['y']
            
            # Create transformed item
            transformed_item = {
                'label': annotation['name'],
                'confidence': annotation['score'],
                'x': top_left['x'],
                'y': top_left['y'],
                'boxWidth': box_width,
                'boxHeight': box_height,
            }
            
            transformed_items.append(transformed_item)
        
        # Replace with transformed items
        response['items'] = transformed_items
        # Optionally keep or remove the original annotations
        del response['localizedObjectAnnotations']
    
    return response_data

def get_base64_from_url(url):
    """
    Fetch an image from a URL and convert it to base64 encoding
    
    Args:
        url: URL of the image to download
        
    Returns:
        str: Base64-encoded image data
    """
    logger.info(f"Fetching image from URL for base64 encoding: {url}")
    
    # Check if this is an S3 URL (which should be more reliable than external services)
    is_s3_url = "s3.amazonaws.com" in url or "cloudfront.net" in url
    if is_s3_url:
        logger.info("✅ S3/CloudFront URL detected - these should be reliable")
    
    try:
        # Use a browser-like user agent to avoid potential blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Add a retry mechanism for non-S3 URLs that might have rate limiting
        max_retries = 1 if is_s3_url else 3
        retry_delay = 1.0
        
        for retry_count in range(max_retries):
            try:
                # Fetch the image with a timeout
                response = requests.get(url, headers=headers, timeout=10)
                
                # Check for rate limiting
                if response.status_code == 429:
                    if retry_count < max_retries - 1:
                        # Only log a warning if we still have retries left
                        logger.warning(f"Rate limit (429) encountered on attempt {retry_count+1}, retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        # On the last retry, raise the exception
                        logger.error(f"Rate limit (429) still encountered after {max_retries} attempts")
                        response.raise_for_status()
                else:
                    # If not a 429, check for other status codes
                    response.raise_for_status()
                    break
            except requests.exceptions.RequestException as e:
                if retry_count < max_retries - 1:
                    logger.warning(f"Request error on attempt {retry_count+1}: {str(e)}, retrying...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Request failed after {max_retries} attempts: {str(e)}")
                    raise
        
        # Check if we got an image
        content_type = response.headers.get('Content-Type', '')
        logger.info(f"Received response: status={response.status_code}, content-type={content_type}, size={len(response.content)} bytes")
        
        if not content_type.startswith('image/'):
            if is_s3_url:
                # S3 sometimes returns incorrect content types, try to proceed anyway
                logger.warning(f"S3 URL did not return an image Content-Type, but proceeding anyway: {content_type}")
            else:
                logger.warning(f"URL did not return an image. Content-Type: {content_type}")
        
        # Process the image bytes
        image_bytes = BytesIO(response.content)
        encoded = base64.b64encode(image_bytes.read()).decode('utf-8')
        logger.info(f"Successfully encoded image, base64 length: {len(encoded)}")
        return encoded
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed when fetching image: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"Error encoding image to base64: {str(e)}")
        raise

@csrf_exempt
def detect_objects(request):
    """
    Process image and detect objects using Google Vision API.
    Expected request format: JSON with either 'image_data' or 'product_url'.
    Results are filtered to only include clothing categories.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
    try:
        # Parse request data
        data = json.loads(request.body)
        
        # Check for required fields
        if 'image_data' not in data and 'product_url' not in data:
            return JsonResponse({'error': 'The request must include either image_data or product_url field'}, status=400)
        
        # Get base64 image data
        try:
            if 'image_data' in data:
                image_data = data['image_data']
                
                # Extract the base64 part if the data is in data URL format
                if image_data.startswith('data:image'):
                    base64data = image_data.split(",")[1]
                else:
                    base64data = image_data
            else:
                product_url = data['product_url']
                logger.info(f"Fetching image from URL: {product_url}")
                base64data = get_base64_from_url(product_url)
                logger.info("Successfully fetched and encoded image from URL")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching image from URL: {str(e)}")
            return JsonResponse({'error': f'Error fetching image from URL: {str(e)}'}, status=400)
        
        # Call Vision API
        vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={settings.GOOGLE_VISION_API_KEY}"
        
        vision_request = {
            "requests": [
                {
                    "image": {"content": base64data},
                    "features": [{"type": "OBJECT_LOCALIZATION", "maxResults": 20}]
                }
            ]
        }
        
        vision_response = requests.post(vision_api_url, json=vision_request)
        
        if vision_response.status_code == 200:
            # Process response
            response_data = vision_response.json()
            filtered_data = filter_clothing_items(response_data)
            transformed_data = transform_response(filtered_data)
            return JsonResponse(transformed_data)
        else:
            logger.error(f"Google Vision API error: {vision_response.status_code} - {vision_response.text}")
            return JsonResponse({
                'error': 'Error calling Google Vision API',
                'status_code': vision_response.status_code,
                'details': vision_response.text
            }, status=500)
            
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        logger.exception("Error processing image for object detection")
        return JsonResponse({'error': str(e)}, status=500)
