import logging
import requests
from django.conf import settings
import boto3
from botocore.exceptions import ClientError
import uuid

s3_prefix = "https://feed-images-01.s3.amazonaws.com"
cdn_prefix = "https://d19dlu1w9mnmln.cloudfront.net"

def get_cdn_url(url):
    if not url:
        return None
    print(f"url: {url}")
    return url.replace(s3_prefix, cdn_prefix)

def safe_rehost(url, fallback_url=None):
    logger = logging.getLogger(__name__)
    
    if not url:
        logger.warning("Cannot rehost empty URL")
        return fallback_url or url
    
    try:
        # Import the rehost_image function (to avoid circular imports)
        from instagram_bot.views import rehost_image
        
        # Attempt to rehost the image
        s3_url = rehost_image(url, None)
        
        # If rehosting succeeded, convert to CDN URL
        if s3_url:
            cdn_url = get_cdn_url(s3_url)
            logger.info(f"Successfully rehosted image: {url} → {cdn_url}")
            return cdn_url
        
        # Rehosting failed, use fallback or original
        logger.warning(f"Rehosting failed for {url}, using fallback")
        return fallback_url or url
        
    except Exception as e:
        # Log the error and return fallback URL
        logger.exception(f"Error in safe_rehost: {str(e)}")
        return fallback_url or url

def save_image_to_s3(image_file, subfolder="uploads", bucket_name=None):
    logger = logging.getLogger(__name__)
    
    if not image_file:
        logger.error("No image file provided")
        return None
    
    # Use default bucket if not specified
    if not bucket_name:
        bucket_name = 'feed-images-01'
    
    s3_client = None
    try:
        # Create a unique filename
        if hasattr(image_file, 'name'):
            file_parts = image_file.name.split('.')
            file_extension = file_parts[-1] if len(file_parts) > 1 else 'jpg'
        else:
            file_extension = 'jpg'
            
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Prepare path
        s3_path = f"{subfolder}/{unique_filename}"
        if subfolder == "":
            s3_path = unique_filename
            
        # Extra args for upload
        extra_args = {}
        if hasattr(image_file, 'content_type') and image_file.content_type:
            extra_args['ContentType'] = image_file.content_type
        
        # Upload to S3
        s3_client.upload_fileobj(
            image_file,
            bucket_name,
            s3_path,
            ExtraArgs=extra_args
        )
        
        # Generate the URL
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_path}"
        logger.info(f"Successfully uploaded image to S3: {s3_url}")
        
        return s3_url
        
    except ClientError as e:
        logger.error(f"S3 upload error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during S3 upload: {str(e)}")
        return None
    finally:
        # Clean up resources
        if image_file and hasattr(image_file, 'close'):
            try:
                image_file.close()
            except:
                pass
        
        # Clean up boto3 resources if needed
        if s3_client:
            try:
                s3_client.close()
            except:
                pass

def download_image_from_url(url):
    logger = logging.getLogger(__name__)
    
    if not url:
        logger.error("No URL provided for download")
        return None, None
    
    try:
        # Set up headers for the request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Make the request with streaming to avoid loading everything into memory at once
        with requests.get(url, headers=headers, timeout=10, stream=True) as response:
            # Check if request was successful
            if response.status_code != 200:
                logger.error(f"Failed to download image from URL: {url}, status code: {response.status_code}")
                return None, None
            
            # Get content type
            content_type = response.headers.get('Content-Type', 'image/jpeg')
            if not content_type.startswith('image/'):
                logger.warning(f"URL content type is not an image: {content_type}")
                # Continue anyway as sometimes content type headers are incorrect
            
            # Create a file-like object from the response content
            from io import BytesIO
            image_file = BytesIO(response.content)
            
            # Add name and content_type attributes
            extension = content_type.split('/')[-1]
            if extension == 'jpeg':
                extension = 'jpg'  # Standardize jpeg to jpg
                
            image_file.name = f"downloaded_image.{extension}"
            image_file.content_type = content_type
            
            return image_file, content_type
        
    except Exception as e:
        logger.exception(f"Error downloading image from URL: {str(e)}")
        return None, None

def download_and_save_to_s3(url, subfolder="downloads", bucket_name=None):
    """
    Download an image from a URL and save it to S3.
    
    Args:
        url (str): The URL of the image to download
        subfolder (str): The subfolder in the S3 bucket to store the image
        bucket_name (str, optional): The S3 bucket name
        
    Returns:
        str: The S3 URL of the uploaded image or None if failed
    """
    logger = logging.getLogger(__name__)
    
    if not url:
        logger.error("No URL provided for download and save")
        return None
    
    try:
        # Download the image
        image_file, _ = download_image_from_url(url)
        if not image_file:
            return None
            
        # Save to S3
        return save_image_to_s3(image_file, subfolder=subfolder, bucket_name=bucket_name)
        
    except Exception as e:
        logger.exception(f"Error in download_and_save_to_s3: {str(e)}")
        return None