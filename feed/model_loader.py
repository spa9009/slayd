import torch
from fashion_clip.fashion_clip import FashionCLIP
from transformers import CLIPProcessor, CLIPModel
from threading import Lock
import logging

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
        logger = logging.getLogger(__name__)
        try:
            # Initialize FashionCLIP
            self.fclip = FashionCLIP('fashion-clip')
            
            # Initialize CLIP model and processor
            self.clip_model = CLIPModel.from_pretrained(
                "patrickjohncyh/fashion-clip",
                torch_dtype=torch.float16  # Memory optimization
            )
            self.clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

            # Create normalized dummy image tensor (values between 0 and 1)
            dummy_image = torch.rand(1, 3, 224, 224)  # Using rand instead of randn
            
            # Process dummy inputs
            dummy_inputs = self.clip_processor(
                images=dummy_image,
                text=["a photo of a dress"],
                return_tensors="pt",
                padding=True
            )

            # Move model to eval mode and disable gradients
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False

            # Trace the model with proper inputs
            with torch.no_grad():
                self.clip_model = torch.jit.trace_module(
                    self.clip_model,
                    {
                        'get_image_features': (dummy_inputs['pixel_values'],),
                        'get_text_features': (dummy_inputs['input_ids'], dummy_inputs['attention_mask'])
                    }
                )

            logger.info("Model initialization successful")

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

# Global function to get the singleton instance
def get_model_instance():
    return ModelSingleton()