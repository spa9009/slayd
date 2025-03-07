from django.apps import AppConfig
from django.conf import settings



class FeedConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'feed'

    def ready(self):
        from . import signals
        global fclip
        fclip = settings.FASHIONCLIP_MODEL