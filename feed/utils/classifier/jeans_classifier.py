from .apparel_classifier import ApparelClassifier
from .common_attributes import CommonApparelAttributes
from typing import List, Optional
import logging
import gc
import torch
import traceback

class JeansClassifier(ApparelClassifier):
    """Classifier for jeans with jeans-specific attributes"""

    def __init__(self, model, processor, predicted_apparel="jeans"):
        """Initialize the JeansClassifier
        
        Args:
            model: The classification model
            processor: The image processor
            predicted_apparel: The type of apparel (defaults to "jeans")
        """
        super().__init__(model, processor, predicted_apparel)

    def _initialize_attributes(self):
        """Initialize jeans-specific attributes"""
        self._fits = [
            "Skinny", "Slim", "Straight", "Bootcut", "Flared", "Relaxed"
        ]
        self._rise = [
            "High-Rise", "Mid-Rise", "Low-Rise"
        ]
        self._washes = [
            "Dark Wash", "Medium Wash", "Light Wash", "Black", "Distressed"
        ]
        self._materials = [
            "100% Cotton", "Stretch Denim", "Selvedge", "Raw Denim"
        ]
        self._details = [
            "Small Ripped", "Medium Ripped", "Large Ripped", "Embroidered", "Embroidered with Ripped", "Studded", "Plain", "Printed", "Ripped and Printed", "Cargo Pockets"
        ]
        self._colors = [
            "Indigo", "Blue", "Black", "Grey", "White", "Brown", "Pink", "Red", "maroon"
        ]

    @property
    def fits(self) -> List[str]:
        return self._fits

    @property
    def rise(self) -> List[str]:
        return self._rise

    @property
    def washes(self) -> List[str]:
        return self._washes

    @property
    def materials(self) -> List[str]:
        return self._materials

    @property
    def details(self) -> List[str]:
        return self._details

    @property
    def colors(self) -> List[str]:
        return self._colors

    def generate_description(self, image):
        """Generate enhanced description using transformer"""
        try:
            # Fit classification
            fit_categories = [
                f"a photo of {fit.lower()} fit {self.predicted_apparel}" 
                for fit in self.fits
            ]
            predicted_fit = self._classify_attribute(image, fit_categories, self.fits)

            # Rise classification
            rise_categories = [
                f"a photo of {rise.lower()} {self.predicted_apparel}" 
                for rise in self.rise
            ]
            predicted_rise = self._classify_attribute(image, rise_categories, self.rise)

            # Wash classification
            wash_categories = [
                f"a photo of {wash.lower()} {self.predicted_apparel}" 
                for wash in self.washes
            ]
            predicted_wash = self._classify_attribute(image, wash_categories, self.washes)

            # Material classification
            material_categories = [
                f"a photo of {material.lower()} {self.predicted_apparel}" 
                for material in self.materials
            ]
            predicted_material = self._classify_attribute(image, material_categories, self.materials)

            # Details classification with special handling
            predicted_details = self._classify_attribute(image, self._details, self._details)

            # Color classification
            color_categories = [
                f"a photo of {color.lower()} colored {self.predicted_apparel}" 
                for color in self.colors
            ]
            predicted_color = self._classify_attribute(image, color_categories, self.colors)

            # Build description
            description_parts = [
                predicted_color.lower() if predicted_color else "",
                predicted_wash.lower() if predicted_wash else "",
                predicted_rise.lower() if predicted_rise else "",
                predicted_fit.lower() if predicted_fit else "",
                predicted_material.lower() if predicted_material else "",
                self.predicted_apparel
            ]
            
            # Filter out empty strings
            description_parts = [part for part in description_parts if part]
            
            base_description = "" + " ".join(description_parts)
            
            # Add details if present
            if predicted_details and predicted_details.lower() != "plain":
                base_description += f" with {predicted_details.lower()} details"

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

    # def _classify_design_details(self, image, threshold=0.25):
    #     """
    #     Special classification method for design details:
    #     1. First identifies top 5 potential details
    #     2. Then performs binary yes/no classification for each detail
    #     """
    #     details = [detail for detail in self.details if detail.lower() != "plain"]
        
    #     # Step 1: Initial screening to get top 5 potential details
    #     initial_prompts = [
    #         f"a photo of {self.predicted_apparel} with {detail.lower()} details" 
    #         for detail in details
    #     ]
        
    #     inputs = self.processor(
    #         text=initial_prompts,
    #         images=image,
    #         return_tensors="pt",
    #         padding=True
    #     )
        
    #     if torch.cuda.is_available():
    #         inputs = {k: v.cuda() for k, v in inputs.items()}
        
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #         logits = outputs.logits_per_image[0]
    #         probs = torch.nn.functional.softmax(logits, dim=0)
            
    #         # Get top 5 potential details
    #         top_probs, top_indices = torch.topk(probs, min(5, len(details)))
    #         top_details = [details[idx] for idx in top_indices]

    #         print("\nTop 3 predictions:")
    #         for prob, idx in zip(top_probs, top_indices):
    #             print(f"  {details[idx]}: {prob.item():.2%}")
            
    #         # Step 2: Binary classification for each top detail
    #         confirmed_details = []
            
    #         for detail in top_details:
    #             # Create binary choice prompts
    #             binary_prompts = [
    #                 f"this is a photo of {self.predicted_apparel} clearly showing {detail.lower()} details",  # Positive case
    #                 f"this is a photo of {self.predicted_apparel} without {detail.lower()} details"  # Negative case
    #             ]
                
    #             binary_inputs = self.processor(
    #                 text=binary_prompts,
    #                 images=image,
    #                 return_tensors="pt",
    #                 padding=True
    #             )
                
    #             if torch.cuda.is_available():
    #                 binary_inputs = {k: v.cuda() for k, v in binary_inputs.items()}
                
    #             binary_outputs = self.model(**binary_inputs)
    #             binary_logits = binary_outputs.logits_per_image[0]
    #             binary_probs = torch.nn.functional.softmax(binary_logits, dim=0)
                
    #             has_detail_prob = binary_probs[0].item()
    #             no_detail_prob = binary_probs[1].item()
                
    #             # Decision based on probability comparison and threshold
    #             if has_detail_prob > no_detail_prob and has_detail_prob > threshold:
    #                 confirmed_details.append(detail)
            
    #         if confirmed_details:
    #             return confirmed_details[0]  # Return the first confirmed detail
    #         else:
    #             return "Plain"

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