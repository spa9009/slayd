from django.apps import AppConfig
from fashion_clip.fashion_clip import FashionCLIP
from transformers import CLIPProcessor, CLIPModel
import torch



class FeedConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'feed'

    def ready(self):
        from . import signals
        global fclip, clip_model, clip_processor
        
        # Initialize FashionCLIP
        fclip = FashionCLIP('fashion-clip')
        
        # Initialize CLIP transformer model and processor
        clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        if torch.cuda.is_available():
            clip_model = clip_model.cuda()
        clip_model.eval()