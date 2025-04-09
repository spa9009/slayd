import logging
import gc
import torch
import traceback
from typing import List, Optional
from .apparel_classifier import ApparelClassifier

class DressClassifier(ApparelClassifier):
    """Classifier for dresses with dress-specific attributes"""

    def __init__(self, model, processor, predicted_apparel: str = "dress"):
        super().__init__(model, processor, predicted_apparel)

    def _initialize_attributes(self):
        """Initialize dress-specific attributes"""
        self._print_types = [
            "Floral", "Geometric", "Abstract", "Animal", "Striped", 
            "Polka Dot", "Paisley", "Solid"
        ]
        self._print_color_styles = ["Monochrome", "Multicolor"]
        self._print_sizes = ["Small", "Medium", "Large"]
        self._print_status = ["Printed", "Solid"]
        self._colors = [
            "Red", "Blue", "Green", "Yellow", "Purple", "Pink", "Orange",
            "Brown", "Black", "White", "Grey", "Beige", "Navy", "Maroon"
        ]
        self._materials = [
            "Cotton", "Silk", "Polyester", "Linen", "Satin", "Chiffon",
            "Velvet", "Denim", "Wool", "Jersey", "Lace"
        ]
        self._lengths = ["Mini", "Midi", "Maxi", "Knee Length"]
        self._necklines = [
            "V-neck", "Round Neck", "Square Neck", "Halter", "Off Shoulder",
            "Sweetheart", "Cowl Neck", "Boat Neck"
        ]
        self._sleeves = [
            "Sleeveless", "Short Sleeve", "Long Sleeve", "Cap Sleeve",
            "Three Quarter Sleeve", "Flutter Sleeve", "Bell Sleeve"
        ]
        self._fits = ["Fitted", "A-line", "Loose", "Empire Waist", "Bodycon", "Wrap"]
        self._design_features = [
            "Ruffle", "Pleated", "Tiered", "Ruched", "Embroidered", "Sequined",
            "Beaded", "Belted", "Draped", "Layered", "Smocked", "Plain"
        ]

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

    @property
    def materials(self) -> List[str]:
        return self._materials

    @property
    def lengths(self) -> List[str]:
        return self._lengths

    @property
    def necklines(self) -> Optional[List[str]]:
        return self._necklines

    @property
    def sleeves(self) -> Optional[List[str]]:
        return self._sleeves

    @property
    def fits(self) -> List[str]:
        return self._fits

    @property
    def design_features(self) -> List[str]:
        return self._design_features

    def generate_description(self, image):
        """Generate enhanced description using transformer"""
        self.logger.debug("\nStarting Classification Results:")
        self.logger.debug("="*50)

        try:
            # First determine if the dress is printed or solid
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

            # Basic attribute classifications
            color_categories = [
                f"a photo of {'an' if color[0].lower() in 'aeiou' else 'a'} {color} colored dress" 
                for color in self._colors
            ]
            predicted_color = self._classify_attribute(image, color_categories, self._colors)
            self.logger.debug(f"Color: {predicted_color}")

            length_categories = [f"a photo of a {length.lower()} {self.predicted_apparel}" for length in self._lengths]
            predicted_length = self._classify_attribute(image, length_categories, self._lengths)
            self.logger.debug(f"Length: {predicted_length}")

            fit_categories = [f"a photo of a {fit.lower()} {self.predicted_apparel}" for fit in self._fits]
            predicted_fit = self._classify_attribute(image, fit_categories, self._fits)
            self.logger.debug(f"Fit: {predicted_fit}")

            neckline_categories = [f"a photo of a {self.predicted_apparel} with {neckline.lower()}" for neckline in self._necklines]
            predicted_neckline = self._classify_attribute(image, neckline_categories, self._necklines)
            self.logger.debug(f"Neckline: {predicted_neckline}")

            sleeve_categories = [f"a photo of a {sleeve.lower()} {self.predicted_apparel}" for sleeve in self._sleeves]
            predicted_sleeve = self._classify_attribute(image, sleeve_categories, self._sleeves)
            self.logger.debug(f"Sleeves: {predicted_sleeve}")

            material_categories = [f"a photo of a {material.lower()} {self.predicted_apparel}" for material in self._materials]
            predicted_material = self._classify_attribute(image, material_categories, self._materials)
            self.logger.debug(f"Material: {predicted_material}")

            feature_categories = [f"a photo of a {self.predicted_apparel} with {feature.lower()}" for feature in self._design_features]
            predicted_feature = self._classify_attribute(image, feature_categories, self._design_features)
            self.logger.debug(f"Design Feature: {predicted_feature}")

            # Build description
            description_parts = []
            
            # Only add attributes if they're not None
            if predicted_color:
                description_parts.append(predicted_color.lower())
            if predicted_material:
                description_parts.append(predicted_material.lower())
            if predicted_length:
                description_parts.append(predicted_length.lower())
            if predicted_fit:
                description_parts.append(predicted_fit.lower())
            if predicted_neckline:
                description_parts.append(predicted_neckline.lower())
            if predicted_sleeve:
                description_parts.append(predicted_sleeve.lower())
            
            # Add print information if present
            if print_details:
                description_parts = print_details + description_parts
            
            if not description_parts:  # If no attributes were successfully classified
                return f" {self.predicted_apparel}"
                
            base_description = "" + " ".join(description_parts) + " " + self.predicted_apparel
            
            # Only add feature if it exists
            if predicted_feature:
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