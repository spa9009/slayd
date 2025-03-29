import json
import requests
import re
from django.http import JsonResponse, HttpResponse
from django.views.generic import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
import logging
from datetime import datetime, timedelta
import os
from urllib.parse import quote, urlencode
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from pathlib import Path
import uuid
from django.core.cache import cache

logger = logging.getLogger(__name__)

@method_decorator(csrf_exempt, name='dispatch')
class MetaWebhookView(View):
    VERIFY_TOKEN = "slayd"
    INSTAGRAM_API_URL = "https://graph.facebook.com/v22.0/me/messages"
    LOG_FILE = os.path.join(settings.BASE_DIR, 'webhook_log.txt')
    IMGUR_CLIENT_ID = "2202f7d1fca273b"
    PINTEREST_APP_ID = "1512605"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"Token available: {'INSTAGRAM_ACCESS_TOKEN' in dir(settings)}")

    def get(self, request, *args, **kwargs):
        logger.info("=== GET REQUEST RECEIVED ===")
        logger.info(f"Query params: {request.GET}")
        
        # Instagram verification
        mode = request.GET.get('hub.mode')
        token = request.GET.get('hub.verify_token')
        challenge = request.GET.get('hub.challenge')
        
        # If this is a verification request
        if mode and token:
            if mode == 'subscribe' and token == self.VERIFY_TOKEN:
                return HttpResponse(challenge)
            return HttpResponse('Forbidden', status=403)

        return HttpResponse("Webhook is working", content_type="text/plain")

    def get_recent_logs(self):
        try:
            # Try to read the last 5 webhook requests from the log file
            logs = []
            try:
                with open(self.LOG_FILE, 'r') as f:
                    logs = f.readlines()[-5:]  # Get last 5 lines
            except FileNotFoundError:
                logs = ["No webhook logs found yet"]
            except Exception as e:
                logs = [f"Error reading logs: {str(e)}"]
            
            return {
                'recent_requests': logs,
                'log_count': len(logs)
            }
        except Exception as e:
            return {
                'error': str(e),
                'message': 'Error retrieving logs'
            }

    def log_webhook_request(self, data):
        try:
            timestamp = datetime.now().isoformat()
            log_entry = f"\n=== Webhook Request at {timestamp} ===\n{json.dumps(data, indent=2)}\n"
            
            os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)
            with open(self.LOG_FILE, 'a') as f:
                f.write(log_entry)
                
        except Exception as e:
            logger.exception("Error logging request")

    def proxy_facebook_image(self, url):
        """Convert Facebook CDN URL to direct JPEG response"""
        try:
            # Download image from Facebook CDN
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                # Create HTTP response with JPEG content
                img_response = HttpResponse(response.content, content_type="image/jpeg")
                img_response["Content-Disposition"] = "inline; filename=image.jpg"
                return img_response
            else:
                return HttpResponse(f"Failed to fetch image: {response.status_code}", status=400)
        except Exception as e:
            logger.exception("Error proxying image")
            return HttpResponse(f"Error: {str(e)}", status=500)

    def rehost_image(self, url):
        """Rehost image to Imgur"""
        try:
            # Get image from URL
            response = requests.get(url)
            if response.status_code != 200:
                logger.error(f"Failed to fetch image: {response.status_code}")
                return None
            
            # Upload to Imgur
            imgur_url = "https://api.imgur.com/3/image"
            headers = {"Authorization": f"Client-ID {self.IMGUR_CLIENT_ID}"}
            files = {"image": response.content}
            
            logger.info("Uploading to Imgur...")
            upload_response = requests.post(imgur_url, headers=headers, files=files)
            
            if upload_response.status_code == 200:
                imgur_link = upload_response.json()["data"]["link"]
                logger.info(f"Image rehosted successfully: {imgur_link}")
                return imgur_link
            
            logger.error(f"Imgur upload failed: {upload_response.text}")
            return None
            
        except Exception as e:
            logger.exception("Error rehosting image")
            return None

    def get_carousel_image_url(self, url):
        """Extract the specific image URL from carousel share"""
        try:
            # Parse the URL to get asset_id
            if 'asset_id=' in url:
                # Extract asset_id from the URL
                asset_id = url.split('asset_id=')[1].split('&')[0]
                logger.info(f"🎯 Found asset ID: {asset_id}")
                
                # Make request to Instagram Graph API to get the specific image
                api_url = f"https://graph.instagram.com/v12.0/{asset_id}?fields=media_url&access_token={settings.INSTAGRAM_ACCESS_TOKEN}"
                response = requests.get(api_url)
                
                if response.status_code == 200:
                    media_url = response.json().get('media_url')
                    if media_url:
                        logger.info(f"✅ Found specific carousel image: {media_url}")
                        return media_url
            
            logger.info("⚠️ Using original URL as fallback")
            return url
        except Exception as e:
            logger.exception("Error getting carousel image")
            return url

    def handle_carousel_post(self, url, sender_id):
        """Handle carousel post and identify the specific shared image"""
        try:
            # Extract media ID from URL
            media_id_match = re.search(r'media_id=([^&]+)', url)
            if not media_id_match:
                logger.warning("❌ Could not extract media ID from URL")
                return None, "error_invalid_url"

            media_id = media_id_match.group(1)
            logger.info(f"🎯 Processing media with ID: {media_id}")
            
            # Get media using Graph API
            api_url = f"https://graph.instagram.com/v12.0/{media_id}?fields=media_type,media_url,children{{media_url,id}}&access_token={settings.INSTAGRAM_ACCESS_TOKEN}"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"📊 API Response: {json.dumps(data, indent=2)}")
                
                # Handle single image post
                if 'media_url' in data:
                    logger.info(f"✅ Found single image URL: {data['media_url']}")
                    return data['media_url'], None
                
                # Handle carousel post
                if data.get('media_type') == 'CAROUSEL_ALBUM' and 'children' in data:
                    children = data['children']['data']
                    logger.info(f"📑 Found {len(children)} images in carousel")
                    
                    # Get first image URL from carousel
                    if children and 'media_url' in children[0]:
                        logger.info(f"✅ Using first carousel image: {children[0]['media_url']}")
                        return children[0]['media_url'], None
            
            # Check for specific error messages
            error_data = response.json() if response.status_code != 200 else {}
            error_message = error_data.get('error', {}).get('message', '')
            
            if response.status_code == 400 and ('permission' in error_message.lower() or 'private' in error_message.lower()):
                logger.warning("❌ Private post detected")
                return None, "error_private_post"
            
            logger.warning(f"❌ API request failed: {response.status_code} - {response.text}")
            return None, "error_api_failed"
            
        except Exception as e:
            logger.exception("Error handling carousel")
            return None, "error_general"

    def extract_carousel_image(self, attachment):
        """Extract the specific image URL from a carousel share"""
        try:
            payload = attachment.get('payload', {})
            url = payload.get('url')
            
            logger.info("\n🔍 Processing shared post URL:")
            logger.info(f"URL: {url}")
            
            # Handle direct CDN URLs from Instagram
            if 'lookaside.fbsbx.com/ig_messaging_cdn' in url:
                # Extract asset_id from the URL
                asset_id_match = re.search(r'asset_id=([^&]+)', url)
                if asset_id_match:
                    asset_id = asset_id_match.group(1)
                    logger.info(f"📍 Found asset ID: {asset_id}")
                    
                    # Make request to Instagram Graph API to get the image
                    api_url = f"https://graph.instagram.com/v12.0/{asset_id}?fields=media_url&access_token={settings.INSTAGRAM_ACCESS_TOKEN}"
                    response = requests.get(api_url)
                    
                    if response.status_code == 200:
                        media_url = response.json().get('media_url')
                        if media_url:
                            logger.info(f"✅ Found image URL: {media_url}")
                            return media_url, None
                        
                    error_data = response.json() if response.status_code != 200 else {}
                    error_message = error_data.get('error', {}).get('message', '')
                    
                    if response.status_code == 400 and ('permission' in error_message.lower() or 'private' in error_message.lower()):
                        return None, "error_private_post"
                        
                    logger.warning(f"⚠️ Could not get media URL from API: {response.text}")
                    return None, "error_api_failed"
                
                logger.warning("⚠️ Using original CDN URL")
                return url, None
            
            # Handle regular Instagram post shares
            elif 'instagram.com' in url:
                media_id_match = re.search(r'media_id=([^&]+)', url)
                if media_id_match:
                    media_id = media_id_match.group(1)
                    return self.handle_carousel_post(url, None)
            
            logger.warning("⚠️ Using original URL as fallback")
            return url, None
            
        except Exception as e:
            logger.exception("Error processing image")
            return None, "error_general"

    def extract_pinterest_image(self, text):
        """Extract image URL from Pinterest link using Pinterest API"""
        try:
            logger.info(f"\n🔍 Processing text for Pinterest URL: {text}")
            
            # Extract pin ID from URL
            pin_id = None
            
            # Check for short URL
            short_urls = re.findall(r'https?://pin\.it/\w+', text)
            if short_urls:
                pin_url = short_urls[0]
                logger.info(f"📌 Found short URL: {pin_url}")
                # Resolve the short URL
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(pin_url, headers=headers, allow_redirects=True)
                pin_url = response.url
                pin_match = re.search(r'/pin/(\d+)', pin_url)
                if pin_match:
                    pin_id = pin_match.group(1)
            else:
                # Updated pattern to match various Pinterest domains
                pinterest_pattern = r'https?://(?:[a-z]{2,3}\.)?pinterest\.(?:com|[a-z]{2,3})/pin/(\d+)'
                pin_match = re.search(pinterest_pattern, text)
                if pin_match:
                    pin_id = pin_match.group(1)

            if not pin_id:
                logger.warning("❌ Could not extract pin ID from URL")
                return None

            logger.info(f"📍 Found Pin ID: {pin_id}")
            
            # Make Pinterest API request
            api_url = f"https://api.pinterest.com/v5/pins/{pin_id}"
            headers = {
                "Authorization": f"Bearer {settings.PINTEREST_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"🔄 Making API request to: {api_url}")
            response = requests.get(api_url, headers=headers)
            
            if response.status_code == 200:
                pin_data = response.json()
                logger.info(f"✅ Pinterest API response: {json.dumps(pin_data, indent=2)}")
                
                # Get the highest resolution image URL
                if 'media' in pin_data and 'images' in pin_data['media']:
                    images = pin_data['media']['images']
                    # Try to get the highest resolution (1200x, 600x, 400x300, 150x150)
                    for size in ['1200x', '600x', '400x300', '150x150']:
                        if size in images:
                            image_url = images[size]['url']
                            logger.info(f"✅ Found image URL: {image_url}")
                            return image_url
                
                logger.warning("❌ No image URLs found in API response")
                # Fallback to HTML scraping if API doesn't return image
                return self._fallback_scrape_image(pin_id)
            
            elif response.status_code == 403:
                logger.warning("❌ Not authorized to access Pin. Falling back to scraping.")
                return self._fallback_scrape_image(pin_id)
            elif response.status_code == 404:
                logger.warning("❌ Pin not found")
                return self._fallback_scrape_image(pin_id)
            else:
                logger.error(f"❌ Pinterest API error: {response.status_code} - {response.text}")
                return self._fallback_scrape_image(pin_id)
            
        except Exception as e:
            logger.exception("Error extracting Pinterest image")
            return None

    def _fallback_scrape_image(self, pin_id):
        """Fallback method to scrape image directly from Pinterest"""
        try:
            logger.info(f"🔄 Attempting fallback scraping for pin {pin_id}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'DNT': '1'
            }
            
            # Try multiple URL patterns
            urls_to_try = [
                f"https://www.pinterest.com/pin/{pin_id}/",
                f"https://in.pinterest.com/pin/{pin_id}/",
                f"https://pinterest.com/pin/{pin_id}/"
            ]
            
            for url in urls_to_try:
                try:
                    response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        html_content = response.text
                        # Try multiple patterns to find the image
                        patterns = [
                            r'<meta property="og:image"\s+content="([^"]+)"',
                            r'<meta name="twitter:image:src"\s+content="([^"]+)"',
                            r'"image_url":"([^"]+)"',
                            r'https://i\.pinimg\.com/originals/[^"]+\.jpg'
                        ]
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, html_content)
                            if matches:
                                image_url = matches[0]
                                if isinstance(image_url, tuple):
                                    image_url = image_url[0]
                                image_url = image_url.replace('\\/', '/')
                                logger.info(f"✅ Found image URL through scraping: {image_url}")
                                return image_url
                except Exception as e:
                    logger.warning(f"⚠️ Error trying URL {url}: {str(e)}")
                    continue
                
            logger.warning("❌ Could not find image through scraping")
            return None
            
        except Exception as e:
            logger.exception("Error in fallback scraping")
            return None

    def test_pinterest_api(self):
        """Test Pinterest API access"""
        try:
            headers = {
                "Authorization": f"Bearer {settings.PINTEREST_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }
            
            # Test with user's boards endpoint
            api_url = "https://api.pinterest.com/v5/user_account"
            logger.info("\n🔍 Testing Pinterest API...")
            logger.info(f"URL: {api_url}")
            logger.info(f"Headers: {headers}")
            
            response = requests.get(api_url, headers=headers)
            logger.info(f"Status Code: {response.status_code}")
            logger.info(f"Response: {response.text}")
            
            return response.status_code == 200
        except Exception as e:
            logger.exception("Pinterest API test failed")
            return False

    def is_human_response_active(self, sender_id):
        """Check if there's an active human response window for this user"""
        cache_key = f"human_response_{sender_id}"
        last_human_response = cache.get(cache_key)
        return bool(last_human_response)

    def mark_human_response(self, sender_id):
        """Mark that a human has responded to this user"""
        cache_key = f"human_response_{sender_id}"
        # Set cache with 1 hour expiration
        cache.set(cache_key, datetime.now(), timeout=3600)  # 3600 seconds = 1 hour

    def handle_text_message(self, sender_id, text, is_echo=False, from_page_id=None):
        """Handle different types of text messages"""
        try:
            # If this is a message from the page (human response)
            if is_echo and from_page_id:
                self.mark_human_response(sender_id)
                return

            # Check if we're in human response window
            if self.is_human_response_active(sender_id):
                # Skip automated responses for text messages
                # but still process Pinterest links
                if 'pin.it/' in text or 'pinterest.com/pin/' in text:
                    self.send_instagram_reply(
                        sender_id,
                        "Processing your Pinterest link... 🔍"
                    )
                    # Extract and process Pinterest image
                    image_url = self.extract_pinterest_image(text)
                    if image_url:
                        imgur_url = self.rehost_image(image_url)
                        if imgur_url:
                            self.send_product_card(
                                recipient_id=sender_id,
                                image_url=imgur_url,
                                title="Found similar products from Pinterest! 🛍️",
                                subtitle="Check out these matches for your Pinterest inspiration"
                            )
                        else:
                            self.send_instagram_reply(
                                sender_id,
                                "Sorry, I had trouble processing that Pinterest image. Could you try sharing another one? 🙏"
                            )
                    return

            # Regular automated response flow
            if 'pin.it/' in text or 'pinterest.com/pin/' in text:
                # ... existing Pinterest handling ...
                self.send_instagram_reply(
                    sender_id,
                    "Processing your Pinterest link... 🔍"
                )
                # Extract and process Pinterest image
                image_url = self.extract_pinterest_image(text)
                if image_url:
                    imgur_url = self.rehost_image(image_url)
                    if imgur_url:
                        self.send_product_card(
                            recipient_id=sender_id,
                            image_url=imgur_url,
                            title="Found similar products from Pinterest! 🛍️",
                            subtitle="Check out these matches for your Pinterest inspiration"
                        )
                    else:
                        self.send_instagram_reply(
                            sender_id,
                            "Sorry, I had trouble processing that Pinterest image. Could you try sharing another one? 🙏"
                        )
                return
            elif self.is_feedback_message(text):
                # ... existing feedback handling ...
                return
            else:
                # Default response for other text messages
                self.send_instagram_reply(
                    sender_id,
                    "Hi! 👋 I'm your fashion assistant and I can help you find similar products from any fashion inspiration!\n\n"
                    "Just share with me:\n"
                    "📸 A photo\n"
                    "🎯 An Instagram post\n"
                    "📌 A Pinterest link\n"
                    "🎬 A video/reel\n\n"
                    "Try it now! Share something you'd like to find. 😊"
                )

        except Exception as e:
            logger.exception("Error handling text message")
            self.send_instagram_reply(
                sender_id,
                "Sorry, I encountered an error. Please try again! 🙏"
            )

    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            logger.info("=== WEBHOOK REQUEST DETAILS ===")
            logger.info(json.dumps(data, indent=2))
            
            self.log_webhook_request(data)
            
            for entry in data.get('entry', []):
                for messaging in entry.get('messaging', []):
                    sender_id = messaging.get('sender', {}).get('id')
                    message = messaging.get('message', {})
                    
                    # Get echo and page ID information
                    is_echo = message.get('is_echo', False)
                    from_page_id = None
                    if is_echo:
                        from_page_id = messaging.get('sender', {}).get('id')
                    
                    # Handle text messages
                    if 'text' in message:
                        text = message.get('text', '').strip()
                        self.handle_text_message(
                            sender_id, 
                            text, 
                            is_echo=is_echo, 
                            from_page_id=from_page_id
                        )
                    
                    # Handle attachments (images, shares, etc.)
                    if 'attachments' in message:
                        # Process attachments regardless of human response window
                        for attachment in message.get('attachments', []):
                            attachment_type = attachment.get('type')
                            url = attachment.get('payload', {}).get('url')
                            
                            # Initial response based on type
                            type_responses = {
                                'image': "Processing your image... 🔍",
                                'share': "Processing your shared post... 🔍",
                                'video': "Processing your video... Let me help you find products from it! 🎬",
                                'ig_reel': "Processing your reel... Let me help you find products from it! 🎬"
                            }
                            
                            if attachment_type in type_responses:
                                self.send_instagram_reply(
                                    sender_id,
                                    type_responses[attachment_type]
                                )
                            
                            # Handle different attachment types
                            if attachment_type == 'image':
                                imgur_url = self.rehost_image(url)
                                if imgur_url:
                                    self.send_product_card(
                                        recipient_id=sender_id,
                                        image_url=imgur_url,
                                        title="Found similar products! 🛍️",
                                        subtitle="Check out these matches for your style"
                                    )
                                else:
                                    self.send_instagram_reply(
                                        sender_id,
                                        "Sorry, I had trouble processing that image. Could you try sharing another one? 🙏"
                                    )
                                    
                            elif attachment_type == 'share':
                                # Inform about carousel limitation
                                self.send_instagram_reply(
                                    sender_id,
                                    "Note: For carousel posts, I can only process the first photo due to Instagram restrictions. If you want to find products from other photos in the carousel, please take a screenshot and share it with me! 📸"
                                )
                                
                                url, error_type = self.extract_carousel_image(attachment)
                                
                                # Handle different error types
                                if error_type == "error_private_post":
                                    self.send_instagram_reply(
                                        sender_id,
                                        "Sorry, I can't process posts from private accounts! 🔒\n\n"
                                        "Please make sure to:\n"
                                        "1. Share posts from public accounts only\n"
                                        "2. Or take a screenshot of the post and share it with me\n"
                                        "3. Or share a different inspiration from a public account 📸"
                                    )
                                    return
                                elif error_type == "error_api_failed":
                                    self.send_instagram_reply(
                                        sender_id,
                                        "Sorry, I had trouble accessing that post. Could you try sharing a screenshot instead? 📸"
                                    )
                                    return
                                elif error_type == "error_general":
                                    self.send_instagram_reply(
                                        sender_id,
                                        "Sorry, something went wrong. Could you try sharing the post again or send a screenshot? 📸"
                                    )
                                    return
                                
                                if url:
                                    imgur_url = self.rehost_image(url)
                                    if imgur_url:
                                        self.send_product_card(
                                            recipient_id=sender_id,
                                            image_url=imgur_url,
                                            title="Found similar products! 🛍️",
                                            subtitle="Check out these matches for your style"
                                        )
                                    else:
                                        self.send_instagram_reply(
                                            sender_id,
                                            "Sorry, I had trouble processing that shared post. Could you try sharing a screenshot instead? 📸"
                                        )
                                    
                            elif attachment_type in ['video', 'ig_reel']:
                                try:
                                    # Extract thumbnail or first frame URL if available
                                    thumbnail_url = (
                                        attachment.get('payload', {}).get('thumbnail_url') or 
                                        attachment.get('payload', {}).get('url')
                                    )
                                    
                                    if thumbnail_url:
                                        # Try to process the thumbnail
                                        imgur_url = self.rehost_image(thumbnail_url)
                                        if imgur_url:
                                            self.send_product_card(
                                                recipient_id=sender_id,
                                                image_url=imgur_url,
                                                title="Found products from your video! 🎬",
                                                subtitle="Check out these matches from your video"
                                            )
                                            return
                                    
                                    # If no thumbnail or processing failed
                                    self.send_instagram_reply(
                                        sender_id,
                                        "I can process your video/reel! Please:\n\n"
                                        "1. Pause the video at the product you're interested in 🎯\n"
                                        "2. Take a screenshot 📸\n"
                                        "3. Share the screenshot with me\n\n"
                                        "I'll find similar products for you! ✨"
                                    )
                                except Exception as e:
                                    logger.exception("Error processing video/reel")
                                    self.send_instagram_reply(
                                        sender_id,
                                        "I can help you find products from your video/reel! Please:\n\n"
                                        "1. Pause at the product you like 🎯\n"
                                        "2. Take a screenshot 📸\n"
                                        "3. Share it with me\n\n"
                                        "I'll find similar products for you! ✨"
                                    )

            return JsonResponse({'status': 'success'})
                
        except Exception as e:
            error_msg = f"Error processing webhook: {str(e)}"
            logger.exception(error_msg)
            return JsonResponse({'error': str(e)}, status=500)

    def send_instagram_reply(self, recipient_id, message_text):
        try:
            logger.info(f"\n🚀 Attempting to send reply:")
            logger.info(f"To: {recipient_id}")
            logger.info(f"Message: {message_text}")
            
            url = f"https://graph.instagram.com/v22.0/17841472211809579/messages"
            headers = {
                "Authorization": f"Bearer {settings.INSTAGRAM_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {
                "recipient": {"id": recipient_id},
                "message": {"text": message_text}
            }
            
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                raise Exception(f"API error: {response.text}")
            
            logger.info("✅ Message sent successfully!")
            return True
            
        except Exception as e:
            logger.exception("Error sending reply")
            return False

    def send_product_card(self, recipient_id, image_url, title, subtitle):
        try:
            url = f"https://graph.instagram.com/v22.0/17841472211809579/messages"
            headers = {
                "Authorization": f"Bearer {settings.INSTAGRAM_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }
            
            encoded_url = quote(image_url)
            payload = {
                "recipient": {"id": recipient_id},
                "message": {
                    "attachment": {
                        "type": "template",
                        "payload": {
                            "template_type": "generic",
                            "elements": [{
                                "title": title,
                                "image_url": image_url,
                                "subtitle": subtitle,
                                "buttons": [{
                                    "type": "web_url",
                                    "url": f"https://slayd.in/similar-product/?image_url={encoded_url}",
                                    "title": "View Similar Products"
                                }]
                            }]
                        }
                    }
                }
            }
            
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                raise Exception(f"API error: {response.text}")
            
            return True
            
        except Exception as e:
            logger.exception("Error sending product card")
            return False