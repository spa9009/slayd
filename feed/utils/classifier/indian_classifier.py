from .apparel_classifier import ApparelClassifier
from typing import List
import logging
import gc
import torch
import traceback

class IndianClassifier(ApparelClassifier):
    """Classifier for Indian ethnic wear with specific attributes"""

    def __init__(self, model, processor, predicted_apparel="Indian Ethnic"):
        """Initialize the IndianClassifier
        
        Args:
            model: The classification model
            processor: The image processor
            predicted_apparel: The type of apparel (defaults to "Indian Ethnic")
        """
        super().__init__(model, processor, predicted_apparel)
        self._initialize_attributes()

    def _initialize_attributes(self):
        """Initialize Indian ethnic wear specific attributes"""
        self._styles = [
            "Printed", "Embroidered", "Solid"
        ]
        self._colors = [
            "Orange", "Red", "Green", "Grey", "Pink", 
            "Blue", "Purple", "White", "Black", "Yellow", "Beige",
            "Maroon", "Burgundy", "Brown", "Golden"
        ]

    @property
    def styles(self) -> List[str]:
        return self._styles

    @property
    def colors(self) -> List[str]:
        return self._colors

    def _classify_style(self, image):
        """Determine if the item is solid, printed, or embroidered"""
        style_categories = [
            f"a photo of a {style.lower()} {self.predicted_apparel}" 
            for style in self.styles
        ]
        
        predicted_style = self._classify_attribute(image, style_categories, self.styles)
        self.logger.debug(f"Style: {predicted_style}")
        return predicted_style

    def _classify_color(self, image):
        """Determine the color of the item"""
        color_categories = [
            f"a photo of {'an' if x[0].lower() in 'aeiou' else 'a'} {x} colored {self.predicted_apparel}" 
            for x in self.colors
        ]
        
        predicted_color = self._classify_attribute(image, color_categories, self.colors)
        self.logger.debug(f"Color: {predicted_color}")
        return predicted_color

    def generate_description(self, image):
        """
        Generate a description for Indian ethnic wear.
        
        Args:
            image: The input image to classify
            
        Returns:
            str: A description with color and style (solid/printed/embroidered)
        """
        try:
            # Style classification
            predicted_style = self._classify_style(image)
            
            # Color classification
            predicted_color = self._classify_color(image)
            
            # Build description
            description_parts = [
                predicted_color.lower() if predicted_color else "",
                predicted_style.lower() if predicted_style else "",
            ]
            
            # Filter out empty strings
            description_parts = [part for part in description_parts if part]
            
            return " " + " ".join(description_parts)

        except Exception as e:
            self.logger.error(f"Error in generate_description: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _classify_attribute(self, image, categories, attribute_list):
        """Helper method for zero-shot classification using transformer"""
        inputs = self.processor(
            text=categories,
            images=image,
            return_tensors="pt",
            padding=True
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            
            # Get top 3 predictions and their probabilities
            top_probs, top_indices = torch.topk(probs[0], min(3, len(attribute_list)))
            
            # Print top 3 predictions with their probabilities
            print("\nTop 3 predictions:")
            for prob, idx in zip(top_probs, top_indices):
                print(f"  {attribute_list[idx]}: {prob.item():.2%}")
            
            predicted_idx = top_indices[0].item()
            return attribute_list[predicted_idx]
