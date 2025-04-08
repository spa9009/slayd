from .apparel_classifier import ApparelClassifier
from typing import List, Optional
import logging
import gc
import torch
import traceback

class PantClassifier(ApparelClassifier):
    """Classifier for pants with pants-specific attributes"""

    def __init__(self, model, processor, predicted_apparel="pants"):
        """Initialize the PantClassifier
        
        Args:
            model: The classification model
            processor: The image processor
            predicted_apparel: The type of apparel (defaults to "pants")
        """
        super().__init__(model, processor, predicted_apparel)

    def _initialize_attributes(self):
        """Initialize pants-specific attributes"""
        # Print-related attributes
        self._print_types = [
            "Floral", "Geometric", "Abstract", "Animal", "Striped", 
            "Polka Dot", "Paisley", "Solid", "Plaid", "Checkered",
            "Camouflage"
        ]
        self._print_color_styles = ["Monochrome", "Multicolor"]
        self._print_sizes = ["Small", "Medium", "Large"]
        self._print_status = ["Printed", "Solid"]

        # Tier 1: Pants Type
        self._types = [
            'Jeans', 'Chinos', 'Dress Pants', 'Joggers', 
            'Cargo Pants', 'Leggings', 'Slacks', 'Pleated Pants',
            'Harem Pants', 'Capris', 'Culottes', 'Palazzo', 'Sweatpants', 'Shorts', 'mini shorts', 'denim shorts', 'cargo shorts', 'long baggy shorts', 'booty shorts'
        ]

        # Tier 2: Core Attributes
        self._fits = [
            'Skinny', 'Slim', 'Straight', 'Relaxed', 
            'Loose', 'Oversized', 'Tapered', 'Slim & Flared'
        ]
        self._lengths = [
            'Cropped', 'Ankle-Length', 'Standard', 
            'Tall', 'Floor-Length'
        ]
        self._materials = [
            'Denim', 'Cotton', 'Polyester', 'Wool', 
            'Linen', 'Silk', 'Leather', 'Spandex Blend'
        ]

        # Tier 3: Visual Features
        self._design_features = [
            'Ripped', 'Distressed', 'Cuffed', 'Button Fly',
            'Zipper', 'Pleats', 'Side Stripes', 'Drawstring',
            'Elastic Waist', 'Belt Loops', 'Flared', 'V-Shaped waistband',
        ]
        self._colors = [
            'Black', 'Blue', 'White', 'Grey', 'Beige',
            'Brown', 'Khaki', 'Olive', 'Navy', 'Patterned'
        ]

    @property
    def types(self) -> List[str]:
        return self._types

    @property
    def fits(self) -> List[str]:
        return self._fits

    @property
    def lengths(self) -> List[str]:
        return self._lengths

    @property
    def materials(self) -> List[str]:
        return self._materials

    @property
    def design_features(self) -> List[str]:
        return self._design_features

    @property
    def colors(self) -> List[str]:
        return self._colors

    @property
    def print_types(self) -> List[str]:
        return self._print_types

    @property
    def print_color_styles(self) -> List[str]:
        return self._print_color_styles

    @property
    def print_sizes(self) -> List[str]:
        return self._print_sizes

    @property
    def print_status(self) -> List[str]:
        return self._print_status

    def generate_description(self, image):
        """Generate enhanced description using transformer"""
        try:
            print("\n=== Pants Classification Results ===")
            print("="*35)

            # First determine if the pants are printed or solid
            print("\nAnalyzing Print Status:")
            print("-"*20)
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
                print("\nAnalyzing Print Pattern:")
                print("-"*20)
                print_type_categories = [
                    f"a photo of a {self.predicted_apparel} with {print_type.lower()} print pattern" 
                    for print_type in self._print_types[:-1]  # Exclude 'Solid'
                ]
                predicted_print_type = self._classify_attribute(
                    image, print_type_categories, self._print_types[:-1]
                )
                self.logger.debug(f"Print Type: {predicted_print_type}")

                # Print color style classification
                print("\nAnalyzing Print Color Style:")
                print("-"*20)
                color_style_categories = [
                    f"a photo of a {self.predicted_apparel} with {style.lower()} print colors" 
                    for style in self._print_color_styles
                ]
                predicted_color_style = self._classify_attribute(
                    image, color_style_categories, self._print_color_styles
                )
                self.logger.debug(f"Print Color Style: {predicted_color_style}")

                # Print size classification
                print("\nAnalyzing Print Size:")
                print("-"*20)
                size_categories = [
                    f"a photo of a {self.predicted_apparel} with {size.lower()} print pattern" 
                    for size in self._print_sizes
                ]
                predicted_size = self._classify_attribute(
                    image, size_categories, self._print_sizes
                )
                self.logger.debug(f"Print Size: {predicted_size}")
                
                print_details = [
                    predicted_color_style.lower(),
                    predicted_size.lower(),
                    f"{predicted_print_type.lower()}-printed"
                ]

            # Type classification
            print("\nAnalyzing Pants Type:")
            print("-"*20)
            type_categories = [
                f"a photo of {type.lower()}" 
                for type in self.types
            ]
            predicted_type = self._classify_attribute(image, type_categories, self.types)
            self.logger.debug(f"Type: {predicted_type}")

            # Color classification (only if not printed)
            if not print_details:
                print("\nAnalyzing Color:")
                print("-"*20)
                color_categories = [
                    f"a photo of {color.lower()} colored {self.predicted_apparel}" 
                    for color in self.colors
                ]
                predicted_color = self._classify_attribute(image, color_categories, self.colors)
                self.logger.debug(f"Color: {predicted_color}")
            else:
                predicted_color = None

            # Fit classification
            print("\nAnalyzing Fit:")
            print("-"*20)
            fit_categories = [
                f"a photo of {fit.lower()} fit {self.predicted_apparel}" 
                for fit in self.fits
            ]
            predicted_fit = self._classify_attribute(image, fit_categories, self.fits)
            self.logger.debug(f"Fit: {predicted_fit}")

            # Length classification
            print("\nAnalyzing Length:")
            print("-"*20)
            length_categories = [
                f"a photo of {length.lower()} {self.predicted_apparel}" 
                for length in self.lengths
            ]
            predicted_length = self._classify_attribute(image, length_categories, self.lengths)
            self.logger.debug(f"Length: {predicted_length}")

            # Material classification
            print("\nAnalyzing Material:")
            print("-"*20)
            material_categories = [
                f"a photo of {material.lower()} {self.predicted_apparel}" 
                for material in self.materials
            ]
            predicted_material = self._classify_attribute(image, material_categories, self.materials)
            self.logger.debug(f"Material: {predicted_material}")

            # Design features classification
            print("\nAnalyzing Design Features:")
            print("-"*20)
            predicted_features = self._classify_attribute(image, self._design_features, self._design_features)
            self.logger.debug(f"Design Features: {predicted_features}")

            print("\n=== Final Classification Summary ===")
            print("="*35)
            print(f"Print Status: {predicted_print_status}")
            if predicted_print_status == "Printed":
                print(f"Print Type: {predicted_print_type}")
                print(f"Print Color Style: {predicted_color_style}")
                print(f"Print Size: {predicted_size}")
            print(f"Type: {predicted_type}")
            if not print_details:
                print(f"Color: {predicted_color}")
            print(f"Fit: {predicted_fit}")
            print(f"Length: {predicted_length}")
            print(f"Material: {predicted_material}")
            print(f"Design Features: {predicted_features}")
            print("="*35)

            # Build description
            description_parts = []
            
            # Add print details if present
            if print_details:
                description_parts.extend(print_details)
            else:
                # Only add color for solid pants
                if predicted_color:
                    description_parts.append(predicted_color.lower())
            
            # Add other attributes
            if predicted_material:
                description_parts.append(predicted_material.lower())
            if predicted_length:
                description_parts.append(predicted_length.lower())
            if predicted_fit:
                description_parts.append(predicted_fit.lower())
            description_parts.append(predicted_type.lower())
            
            
            # Filter out empty strings
            description_parts = [part for part in description_parts if part]
            
            base_description = " " + " ".join(description_parts)
            
            # Add design features if present and not None
            if predicted_features and predicted_features.lower() != "none":
                base_description += f" with {predicted_features.lower()}"

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

    # def _classify_design_features(self, image, threshold=0.25):
    #     """
    #     Special classification method for design features:
    #     1. First identifies top 5 potential features
    #     2. Then performs binary yes/no classification to confirm if each feature exists
    #     3. Returns all confirmed features, or 'None' if no features are confirmed
    #     """
    #     features = [feature for feature in self.design_features if feature.lower() != "none"]
        
    #     # Step 1: Initial screening to get top 5 potential features
    #     initial_prompts = [
    #         f"a photo of {self.predicted_apparel} with {feature.lower()}" 
    #         for feature in features
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
            
    #         # Get top 5 potential features
    #         top_probs, top_indices = torch.topk(probs, min(5, len(features)))
    #         top_features = [features[idx] for idx in top_indices]

    #         print("\nInitial Design Feature predictions:")
    #         print("-"*30)
    #         for prob, idx in zip(top_probs[:3], top_indices[:3]):
    #             print(f"  {features[idx]}: {prob.item():.2%}")
            
    #         # Step 2: Binary verification for each feature
    #         confirmed_features = []
            
    #         print("\nBinary verification results:")
    #         print("-"*30)
    #         for feature in top_features:
    #             # Binary verification with clear yes/no prompts
    #             binary_prompts = [
    #                 f"this is a photo of {self.predicted_apparel} that has {feature.lower()}",  # Positive case
    #                 f"this is a photo of {self.predicted_apparel} that does not have {feature.lower()}"  # Negative case
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
                
    #             has_feature_prob = binary_probs[0].item()
    #             no_feature_prob = binary_probs[1].item()
                
    #             print(f"  {feature}: Has feature: {has_feature_prob:.2%}, No feature: {no_feature_prob:.2%}")
                
    #             # If the model is more confident that the feature exists and meets threshold
    #             if has_feature_prob > no_feature_prob and has_feature_prob > threshold:
    #                 confirmed_features.append(feature)
            
    #         if confirmed_features:
    #             print(f"\nConfirmed features: {', '.join(confirmed_features)}")
    #             # Return all confirmed features joined by 'and'
    #             return " and ".join(confirmed_features)
    #         else:
    #             print("\nNo features confirmed")
    #             return "None"

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
            print("-"*30)
            for prob, idx in zip(top_probs, top_indices):
                print(f"  {attribute_list[idx]}: {prob.item():.2%}")
            
            predicted_idx = top_indices[0].item()
            return attribute_list[predicted_idx]
