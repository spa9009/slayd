from django.apps import AppConfig
from .model_loader import get_model_instance

class FeedConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'feed'

    def ready(self):
        from . import signals
        global model_instance

        # Load singleton instance
        model_instance = get_model_instance()