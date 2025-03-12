import torch
from fashion_clip.fashion_clip import FashionCLIP
from transformers import CLIPProcessor, CLIPModel
from threading import Lock

class ModelSingleton:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:  # Thread-safe initialization
                if cls._instance is None:  # Double-check locking
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Load models once
        self.fclip = FashionCLIP('fashion-clip')
        self.clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

        # Freeze model parameters to optimize memory usage
        for param in self.clip_model.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            self.clip_model = self.clip_model.cuda()

        self.clip_model.eval()  # Set model to evaluation mode

# Global function to get the singleton instance
def get_model_instance():
    return ModelSingleton()