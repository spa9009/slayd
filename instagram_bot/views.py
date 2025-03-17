import json
import requests
import re
from django.http import JsonResponse, HttpResponse
from django.views.generic import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
import logging
from datetime import datetime
import os
from urllib.parse import quote, urlencode
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from pathlib import Path
import uuid

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
        print(f"Token available: {'INSTAGRAM_ACCESS_TOKEN' in dir(settings)}")

    def get(self, request, *args, **kwargs):
        print("=== GET REQUEST RECEIVED ===")
        print(f"Query params: {request.GET}")
        
        # Instagram verification
        mode = request.GET.get('hub.mode')
        token = request.GET.get('hub.verify_token')
        challenge = request.GET.get('hub.challenge')
        
        # If this is a verification request
        if mode and token:
            if mode == 'subscribe' and token == self.VERIFY_TOKEN:
                print("✅ Webhook verified!")
                return HttpResponse(challenge)
            print(f"❌ Token verification failed. Expected '{self.VERIFY_TOKEN}', got '{token}'")
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
            print(f"❌ Error logging request: {e}")

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
            print(f"❌ Error proxying image: {e}")
            return HttpResponse(f"Error: {str(e)}", status=500)

    def rehost_image(self, url):
        """Rehost image to Imgur"""
        try:
            # Get image from URL
            response = requests.get(url)
            if response.status_code != 200:
                print(f"❌ Failed to fetch image: {response.status_code}")
                return None
            
            # Upload to Imgur
            imgur_url = "https://api.imgur.com/3/image"
            headers = {"Authorization": f"Client-ID {self.IMGUR_CLIENT_ID}"}
            files = {"image": response.content}
            
            print("🔄 Uploading to Imgur...")
            upload_response = requests.post(imgur_url, headers=headers, files=files)
            
            if upload_response.status_code == 200:
                imgur_link = upload_response.json()["data"]["link"]
                print(f"✅ Image rehosted successfully: {imgur_link}")
                return imgur_link
            
            print(f"❌ Imgur upload failed: {upload_response.text}")
            return None
            
        except Exception as e:
            print(f"❌ Error rehosting image: {e}")
            return None

    def get_carousel_image_url(self, url):
        """Extract the specific image URL from carousel share"""
        try:
            # Parse the URL to get asset_id
            if 'asset_id=' in url:
                # Extract asset_id from the URL
                asset_id = url.split('asset_id=')[1].split('&')[0]
                print(f"🎯 Found asset ID: {asset_id}")
                
                # Make request to Instagram Graph API to get the specific image
                api_url = f"https://graph.instagram.com/v12.0/{asset_id}?fields=media_url&access_token={settings.INSTAGRAM_ACCESS_TOKEN}"
                response = requests.get(api_url)
                
                if response.status_code == 200:
                    media_url = response.json().get('media_url')
                    if media_url:
                        print(f"✅ Found specific carousel image: {media_url}")
                        return media_url
            
            print("⚠️ Using original URL as fallback")
            return url
        except Exception as e:
            print(f"❌ Error getting carousel image: {e}")
            return url

    def handle_carousel_post(self, url, sender_id):
        """Handle carousel post and identify the specific shared image"""
        try:
            # Extract media ID from URL
            media_id_match = re.search(r'media_id=([^&]+)', url)
            if not media_id_match:
                print("❌ Could not extract media ID from URL")
                return url

            media_id = media_id_match.group(1)
            print(f"🎯 Processing media with ID: {media_id}")
            
            # Get media using Graph API
            api_url = f"https://graph.instagram.com/v12.0/{media_id}?fields=media_type,media_url,children{{media_url,id}}&access_token={settings.INSTAGRAM_ACCESS_TOKEN}"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                print(f"📊 API Response: {json.dumps(data, indent=2)}")
                
                # Handle single image post
                if 'media_url' in data:
                    print(f"✅ Found single image URL: {data['media_url']}")
                    return data['media_url']
                
                # Handle carousel post
                if data.get('media_type') == 'CAROUSEL_ALBUM' and 'children' in data:
                    children = data['children']['data']
                    print(f"📑 Found {len(children)} images in carousel")
                    
                    # Get first image URL from carousel
                    if children and 'media_url' in children[0]:
                        print(f"✅ Using first carousel image: {children[0]['media_url']}")
                        return children[0]['media_url']
            
            print(f"❌ API request failed: {response.status_code} - {response.text}")
            return url
            
        except Exception as e:
            print(f"❌ Error handling carousel: {e}")
            return url

    def extract_carousel_image(self, attachment):
        """Extract the specific image URL from a carousel share"""
        try:
            payload = attachment.get('payload', {})
            url = payload.get('url')
            
            print("\n🔍 Processing shared post URL:")
            print(f"URL: {url}")
            
            # Handle direct CDN URLs from Instagram
            if 'lookaside.fbsbx.com/ig_messaging_cdn' in url:
                # Extract asset_id from the URL
                asset_id_match = re.search(r'asset_id=([^&]+)', url)
                if asset_id_match:
                    asset_id = asset_id_match.group(1)
                    print(f"📍 Found asset ID: {asset_id}")
                    
                    # Make request to Instagram Graph API to get the image
                    api_url = f"https://graph.instagram.com/v12.0/{asset_id}?fields=media_url&access_token={settings.INSTAGRAM_ACCESS_TOKEN}"
                    response = requests.get(api_url)
                    
                    if response.status_code == 200:
                        media_url = response.json().get('media_url')
                        if media_url:
                            print(f"✅ Found image URL: {media_url}")
                            return media_url
                        
                    print(f"⚠️ Could not get media URL from API: {response.text}")
                    return url
                
                print("⚠️ Using original CDN URL")
                return url
            
            # Handle regular Instagram post shares
            elif 'instagram.com' in url:
                media_id_match = re.search(r'media_id=([^&]+)', url)
                if media_id_match:
                    media_id = media_id_match.group(1)
                    return self.handle_carousel_post(url, None)
            
            print("⚠️ Using original URL as fallback")
            return url
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return url

    def extract_pinterest_image(self, text):
        """Extract image URL from Pinterest link using Pinterest API"""
        try:
            print(f"\n🔍 Processing text for Pinterest URL: {text}")
            
            # Extract pin ID from URL
            pin_id = None
            
            # Check for short URL
            short_urls = re.findall(r'https?://pin\.it/\w+', text)
            if short_urls:
                pin_url = short_urls[0]
                print(f"📌 Found short URL: {pin_url}")
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
                print("❌ Could not extract pin ID from URL")
                return None

            print(f"📍 Found Pin ID: {pin_id}")
            
            # Make Pinterest API request
            api_url = f"https://api.pinterest.com/v5/pins/{pin_id}"
            headers = {
                "Authorization": f"Bearer {settings.PINTEREST_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }
            
            print(f"🔄 Making API request to: {api_url}")
            response = requests.get(api_url, headers=headers)
            
            if response.status_code == 200:
                pin_data = response.json()
                print(f"✅ Pinterest API response: {json.dumps(pin_data, indent=2)}")
                
                # Get the highest resolution image URL
                if 'media' in pin_data and 'images' in pin_data['media']:
                    images = pin_data['media']['images']
                    # Try to get the highest resolution (1200x, 600x, 400x300, 150x150)
                    for size in ['1200x', '600x', '400x300', '150x150']:
                        if size in images:
                            image_url = images[size]['url']
                            print(f"✅ Found image URL: {image_url}")
                            return image_url
                
                print("❌ No image URLs found in API response")
                # Fallback to HTML scraping if API doesn't return image
                return self._fallback_scrape_image(pin_id)
            
            elif response.status_code == 403:
                print("❌ Not authorized to access Pin. Falling back to scraping.")
                return self._fallback_scrape_image(pin_id)
            elif response.status_code == 404:
                print("❌ Pin not found")
                return self._fallback_scrape_image(pin_id)
            else:
                print(f"❌ Pinterest API error: {response.status_code} - {response.text}")
                return self._fallback_scrape_image(pin_id)
            
        except Exception as e:
            print(f"❌ Error extracting Pinterest image: {e}")
            return None

    def _fallback_scrape_image(self, pin_id):
        """Fallback method to scrape image directly from Pinterest"""
        try:
            print(f"🔄 Attempting fallback scraping for pin {pin_id}")
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
                                print(f"✅ Found image URL through scraping: {image_url}")
                                return image_url
                except Exception as e:
                    print(f"⚠️ Error trying URL {url}: {str(e)}")
                    continue
                
            print("❌ Could not find image through scraping")
            return None
            
        except Exception as e:
            print(f"❌ Error in fallback scraping: {e}")
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
            print(f"\n🔍 Testing Pinterest API...")
            print(f"URL: {api_url}")
            print(f"Headers: {headers}")
            
            response = requests.get(api_url, headers=headers)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Pinterest API test failed: {e}")
            return False

    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            print("\n=== WEBHOOK REQUEST DETAILS ===")
            print(json.dumps(data, indent=2))
            
            self.log_webhook_request(data)
            
            for entry in data.get('entry', []):
                for messaging in entry.get('messaging', []):
                    sender_id = messaging.get('sender', {}).get('id')
                    message = messaging.get('message', {})
                    
                    # Skip echo messages
                    if message.get('is_echo'):
                        continue
                    
                    # Handle text messages (for Pinterest links)
                    if 'text' in message:
                        text = message.get('text', '').strip()
                        if 'pin.it/' in text or 'pinterest.com/pin/' in text:
                            print("📌 Pinterest link detected!")
                            self.send_instagram_reply(
                                sender_id,
                                "Processing your Pinterest link... 🔍"
                            )
                            
                            # Extract image from Pinterest
                            image_url = self.extract_pinterest_image(text)
                            if image_url:
                                # Upload to Imgur
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
                            else:
                                self.send_instagram_reply(
                                    sender_id,
                                    "Sorry, I couldn't access that Pinterest pin. Make sure it's a public pin and try again! 🔒"
                                )
                        else:
                            # Handle other text messages as before
                            self.send_instagram_reply(
                                sender_id,
                                "Thanks for your message! 🌟\nBrowse our collection: https://slayd.in/collection"
                            )
                    
                    # Handle attachments (images and shares) as before
                    if 'attachments' in message:
                        for attachment in message.get('attachments', []):
                            attachment_type = attachment.get('type')
                            url = attachment.get('payload', {}).get('url')
                            
                            if attachment_type in ['image', 'share']:
                                if attachment_type == 'share':
                                    url = self.extract_carousel_image(attachment)
                                
                                imgur_url = self.rehost_image(url)
                                if imgur_url:
                                    self.send_product_card(
                                        recipient_id=sender_id,
                                        image_url=imgur_url,
                                        title="Check out these similar products! 🛍️",
                                        subtitle="We found some great matches for your style"
                                    )

            return JsonResponse({'status': 'success'})
                
        except Exception as e:
            error_msg = f"Error processing webhook: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            return JsonResponse({'error': str(e)}, status=500)

    def send_instagram_reply(self, recipient_id, message_text):
        try:
            print(f"\n🚀 Attempting to send reply:")
            print(f"To: {recipient_id}")
            print(f"Message: {message_text}")
            
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
            
            print("✅ Message sent successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            logger.error(f"Failed to send reply: {str(e)}")
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
            print(f"❌ Error: {str(e)}")
            logger.error(f"Failed to send product card: {str(e)}")
            return False