import logging
import gc
import torch
import traceback
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from .common_attributes import ApparelAttributesMixin

class ApparelClassifier(ApparelAttributesMixin, ABC):
    """Abstract base class for apparel classification.
    This class provides the core classification logic but no predefined attributes.
    Each apparel type should implement its own attributes by overriding _initialize_attributes."""
    
    def __init__(self, model, processor, predicted_apparel: str):
        self.model = model
        self.processor = processor
        self.logger = logging.getLogger(__name__)
        self.predicted_apparel = predicted_apparel
        super().__init__()  # Initialize the mixin

    def _classify_attribute(self, image, categories: List[str], attribute_list: List[str]) -> str:
        """Helper method for zero-shot classification using transformer"""
        try:
            if not attribute_list:  # Skip classification if attribute list is empty
                return ""

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
                
                # Log top 3 predictions
                self.logger.debug("\nTop 3 predictions:")
                for prob, idx in zip(top_probs, top_indices):
                    self.logger.debug(f"  {attribute_list[idx]}: {prob.item():.2%}")

                if attribute_list == self.print_status:
                    printed_idx = attribute_list.index("Printed")
                    printed_prob = probs[0][printed_idx].item()
                    if printed_prob >= 0.15: 
                        return "Printed"
                    return "Solid"
                
                predicted_idx = top_indices[0].item()
                return attribute_list[predicted_idx]
                
        except Exception as e:
            self.logger.error(f"Error in _classify_attribute: {str(e)}")
            raise

    def _classify_material(self, image) -> str:
        """Helper method to classify material"""
        material_categories = [
            f"a photo of a {material.lower()} {self.predicted_apparel}" 
            for material in self.materials
        ]
        return self._classify_attribute(image, material_categories, self.materials)

    def _classify_design_feature(self, image) -> str:
        """Helper method to classify design feature"""
        feature_categories = [
            f"a photo of a {self.predicted_apparel} with {feature.lower()}" 
            for feature in self.design_features
        ]
        return self._classify_attribute(image, feature_categories, self.design_features)

    def generate_description(self, image) -> str:
        """Generate enhanced description using transformer"""
        self.logger.debug("\nStarting Classification Results:")
        self.logger.debug("="*50)

        try:
            # First determine if the apparel is printed or solid
            print_status_categories = [
                f"a photo of a {status.lower()} {self.predicted_apparel}" 
                for status in self.print_status
            ]
            predicted_print_status = self._classify_attribute(
                image, print_status_categories, self.print_status
            )
            self.logger.debug(f"Print Status: {predicted_print_status}")

            print_details = []
            if predicted_print_status == "Printed":
                # Print type classification
                print_type_categories = [
                    f"a photo of a {self.predicted_apparel} with {print_type.lower()} print pattern" 
                    for print_type in self.print_types[:-1]  # Exclude 'Solid'
                ]
                predicted_print_type = self._classify_attribute(
                    image, print_type_categories, self.print_types[:-1]
                )
                self.logger.debug(f"Print Type: {predicted_print_type}")

                # Print color style classification
                color_style_categories = [
                    f"a photo of a {self.predicted_apparel} with {style.lower()} print colors" 
                    for style in self.print_color_styles
                ]
                predicted_color_style = self._classify_attribute(
                    image, color_style_categories, self.print_color_styles
                )
                self.logger.debug(f"Print Color Style: {predicted_color_style}")

                # Print size classification
                size_categories = [
                    f"a photo of a {self.predicted_apparel} with {size.lower()} print pattern" 
                    for size in self.print_sizes
                ]
                predicted_size = self._classify_attribute(
                    image, size_categories, self.print_sizes
                )
                self.logger.debug(f"Print Size: {predicted_size}")
                
                print_details = [
                    predicted_color_style.lower(),
                    predicted_size.lower(),
                    f"{predicted_print_type.lower()}-printed"
                ]

            # Basic attribute classifications
            color_categories = [
                f"a photo of {'an' if color[0].lower() in 'aeiou' else 'a'} {color} colored {self.predicted_apparel}" 
                for color in self.colors
            ]
            predicted_color = self._classify_attribute(image, color_categories, self.colors)
            self.logger.debug(f"Color: {predicted_color}")

            length_categories = [f"a photo of a {length.lower()} {self.predicted_apparel}" for length in self.lengths]
            predicted_length = self._classify_attribute(image, length_categories, self.lengths)
            self.logger.debug(f"Length: {predicted_length}")

            fit_categories = [f"a photo of a {fit.lower()} {self.predicted_apparel}" for fit in self.fits]
            predicted_fit = self._classify_attribute(image, fit_categories, self.fits)
            self.logger.debug(f"Fit: {predicted_fit}")

            description_parts = [
                predicted_color.lower(),
                predicted_material.lower() if (predicted_material := self._classify_material(image)) else "",
                predicted_length.lower(),
                predicted_fit.lower(),
            ]

            # Add neckline if applicable
            if self.necklines:
                neckline_categories = [f"a photo of a {self.predicted_apparel} with {neckline.lower()}" for neckline in self.necklines]
                predicted_neckline = self._classify_attribute(image, neckline_categories, self.necklines)
                self.logger.debug(f"Neckline: {predicted_neckline}")
                description_parts.append(predicted_neckline.lower())

            # Add sleeves if applicable
            if self.sleeves:
                sleeve_categories = [f"a photo of a {sleeve.lower()} {self.predicted_apparel}" for sleeve in self.sleeves]
                predicted_sleeve = self._classify_attribute(image, sleeve_categories, self.sleeves)
                self.logger.debug(f"Sleeves: {predicted_sleeve}")
                description_parts.append(predicted_sleeve.lower())

            # Add print information if present
            if print_details:
                description_parts = print_details + description_parts
            
            # Filter out empty strings
            description_parts = [part for part in description_parts if part]
            
            base_description = "This is a " + " ".join(description_parts) + " " + self.predicted_apparel

            # Add design feature if present
            if predicted_feature := self._classify_design_feature(image):
                base_description += f" with {predicted_feature.lower()}"
            
            self.logger.debug("\nFinal Description:")
            self.logger.debug("="*50)
            self.logger.debug(base_description)
            self.logger.debug("="*50)

            return base_description

        except Exception as e:
            self.logger.error(f"Error in generate_description: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect() 