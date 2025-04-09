from .apparel_classifier import ApparelClassifier
from .common_attributes import CommonApparelAttributes
from typing import List, Optional
import logging
import gc
import torch
import traceback

class SkirtClassifier(ApparelClassifier):
    """Classifier for skirts with skirt-specific attributes"""

    def __init__(self, model, processor, predicted_apparel="skirt"):
        """Initialize the SkirtClassifier
        
        Args:
            model: The classification model
            processor: The image processor
            predicted_apparel: The type of apparel (defaults to "skirt")
        """
        super().__init__(model, processor, predicted_apparel)

    def _initialize_attributes(self):
        """Initialize skirt-specific attributes"""
        # Print-related attributes
        self._print_types = [
            "Floral", "Geometric", "Abstract", "Animal", "Striped", 
            "Polka Dot", "Paisley", "Solid"
        ]
        self._print_color_styles = ["Monochrome", "Multicolor"]
        self._print_sizes = ["Small", "Medium", "Large"]
        self._print_status = ["Printed", "Solid"]
        
        # Color attribute
        self._colors = [
            "Red", "Blue", "Green", "Yellow", "Purple", "Pink", "Orange",
            "Brown", "Black", "White", "Grey", "Beige", "Navy", "Maroon"
        ]
        
        # Existing attributes
        self._style = [
            'A-line', 'Pencil', 'Wrap', 'Pleated', 
            'Circle', 'Tulip', 'Bubble', 'Asymmetrical',
            'Denim', 'Cargo', 'Skater', 'Slip'
        ]
        self._length = [
            'Micro (10-15in)', 'Mini (16-20in)', 
            'Above Knee (21-25in)', 'Knee-Length (26-30in)',
            'Midi (31-35in)', 'Maxi (36in+)', 'Floor-Length'
        ]
        self._fit = [
            'Skinny', 'Fitted', 'Semi-Fitted', 
            'Loose', 'Oversized', 'Structured'
        ]
        self._materials = [
            'Cotton', 'Denim', 'Silk', 'Satin', 
            'Chiffon', 'Leather', 'Velvet', 'Tulle',
            'Linen', 'Polyester', 'Mesh', 'Sequin'
        ]
        self._design_features = [
            'Slit', 'Buttons', 'Zipper', 'Ruffles',
            'Lace Trim', 'Belted', 'Pockets', 'Drawstring',
            'Tassels', 'Embroidery', 'Cutouts', 'Sheer Panels',
            'None'
        ]

    # Add new properties for print-related attributes
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

    @property
    def colors(self) -> List[str]:
        return self._colors

    # Existing properties
    @property
    def style(self) -> List[str]:
        return self._style

    @property
    def length(self) -> List[str]:
        return self._length

    @property
    def fit(self) -> List[str]:
        return self._fit

    @property
    def materials(self) -> List[str]:
        return self._materials

    @property
    def design_features(self) -> List[str]:
        return self._design_features

    def generate_description(self, image):
        """Generate enhanced description using transformer"""
        try:
            # First determine if the skirt is printed or solid
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
                    for print_type in self._print_types[:-1]  # Exclude 'Solid'
                ]
                predicted_print_type = self._classify_attribute(
                    image, print_type_categories, self._print_types[:-1]
                )
                self.logger.debug(f"Print Type: {predicted_print_type}")

                # Print color style classification
                color_style_categories = [
                    f"a photo of a {self.predicted_apparel} with {style.lower()} print colors" 
                    for style in self._print_color_styles
                ]
                predicted_color_style = self._classify_attribute(
                    image, color_style_categories, self._print_color_styles
                )
                self.logger.debug(f"Print Color Style: {predicted_color_style}")

                # Print size classification
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

            # Color classification
            color_categories = [
                f"a photo of {'an' if color[0].lower() in 'aeiou' else 'a'} {color.lower()} colored {self.predicted_apparel}" 
                for color in self._colors
            ]
            predicted_color = self._classify_attribute(image, color_categories, self._colors)
            self.logger.debug(f"Color: {predicted_color}")

            # Style classification
            style_categories = [
                f"a photo of {style.lower()} {self.predicted_apparel}" 
                for style in self.style
            ]
            predicted_style = self._classify_attribute(image, style_categories, self.style)
            self.logger.debug(f"Style: {predicted_style}")

            # Length classification
            length_categories = [
                f"a photo of {length.lower()} {self.predicted_apparel}" 
                for length in self.length
            ]
            predicted_length = self._classify_attribute(image, length_categories, self.length)
            self.logger.debug(f"Length: {predicted_length}")

            # Fit classification
            fit_categories = [
                f"a photo of {fit.lower()} {self.predicted_apparel}" 
                for fit in self.fit
            ]
            predicted_fit = self._classify_attribute(image, fit_categories, self.fit)
            self.logger.debug(f"Fit: {predicted_fit}")

            # Material classification
            material_categories = [
                f"a photo of {material.lower()} {self.predicted_apparel}" 
                for material in self.materials
            ]
            predicted_material = self._classify_attribute(image, material_categories, self.materials)
            self.logger.debug(f"Material: {predicted_material}")

            # Design features classification
            predicted_features = self._classify_attribute(image, self._design_features, self._design_features)
            self.logger.debug(f"Design Features: {predicted_features}")

            # Build description
            description_parts = []
            
            # Add print details if present
            if print_details:
                description_parts.extend(print_details)
            else:
                # Only add color for solid skirts
                if predicted_color:
                    description_parts.append(predicted_color.lower())
            
            # Add other attributes if they're not None
            if predicted_material:
                description_parts.append(predicted_material.lower())
            if predicted_length:
                description_parts.append(predicted_length.lower())
            if predicted_fit:
                description_parts.append(predicted_fit.lower())
            if predicted_style:
                description_parts.append(predicted_style.lower())
            
            description_parts.append(self.predicted_apparel)
            
            # Filter out empty strings and join
            description_parts = [part for part in description_parts if part]
            base_description = " " + " ".join(description_parts)
            
            # Add design features if present and not None
            if predicted_features and predicted_features.lower() != "none":
                base_description += f" with {predicted_features.lower()}"

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

    # def _classify_design_features(self, image, threshold=0.25):
    #     """
    #     Special classification method for design features:
    #     1. First identifies top 5 potential features
    #     2. Then performs binary yes/no classification for each feature
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

    #         print("\nTop 3 Design Feature predictions:")
    #         print("-" * 30)
    #         for prob, idx in zip(top_probs[:3], top_indices[:3]):
    #             print(f"  {features[idx]}: {prob.item():.2%}")
            
    #         # Step 2: Binary classification for each top feature
    #         confirmed_features = []
            
    #         for feature in top_features:
    #             # Create binary choice prompts
    #             binary_prompts = [
    #                 f"this is a photo of {self.predicted_apparel} clearly showing {feature.lower()}",  # Positive case
    #                 f"this is a photo of {self.predicted_apparel} without {feature.lower()}"  # Negative case
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
                
    #             # Decision based on probability comparison and threshold
    #             if has_feature_prob > no_feature_prob and has_feature_prob > threshold:
    #                 confirmed_features.append(feature)
            
    #         if confirmed_features:
    #             return confirmed_features[0]  # Return the first confirmed feature
    #         else:
    #             return "None"

    def _classify_attribute(self, image, categories, attribute_list):
        """Helper method for zero-shot classification using transformer"""
        # Get the feature type from the first category
        feature_type = ""
        if "print pattern" in categories[0]:
            feature_type = "Print Pattern"
        elif "print colors" in categories[0]:
            feature_type = "Print Color Style"
        elif "colored" in categories[0]:
            feature_type = "Color"
        elif "length" in categories[0]:
            feature_type = "Length"
        elif "fit" in categories[0]:
            feature_type = "Fit"
        elif "material" in categories[0]:
            feature_type = "Material" 
        elif "design features" in categories[0]:
            feature_type = "Design Features"       
        elif any(style.lower() in categories[0].lower() for style in self.style):
            feature_type = "Style"
        else:
            feature_type = "Attribute"

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
            print(f"\nTop 3 {feature_type} predictions:")
            print("-" * 30)
            for prob, idx in zip(top_probs, top_indices):
                print(f"  {attribute_list[idx]}: {prob.item():.2%}")
            
            predicted_idx = top_indices[0].item()
            return attribute_list[predicted_idx]
