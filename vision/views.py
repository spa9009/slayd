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
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    image_bytes = BytesIO(response.content)
    return base64.b64encode(image_bytes.read()).decode('utf-8')

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
