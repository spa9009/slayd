from django.apps import AppConfig
from fashion_clip.fashion_clip import FashionCLIP



class FeedConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'feed'

    def ready(self):
        from . import signals
        global fclip
        fclip = FashionCLIP('fashion-clip')