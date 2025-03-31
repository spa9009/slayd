from typing import List, Optional

class CommonApparelAttributes:
    """Common attributes that can be shared across different apparel types"""

    @property
    def print_types(self) -> List[str]:
        return [
            "Floral", "Geometric", "Polka Dots", "Leopard", "Zebra",
            "Snake", "Vertical Stripes", "Horizontal Stripes",
            "Diagonal Stripes", "Abstract", "Tie-Dye", "Paisley",
            "Checkered", "Gingham", "Solid"
        ]

    @property
    def print_color_styles(self) -> List[str]:
        return [
            "Pastel", "Bright", "Neon", "Dark", "Ombre",
            "Two-Tone", "Multi-Color"
        ]

    @property
    def print_sizes(self) -> List[str]:
        return [
            "Micro", "Bold", "Delicate", "Large Motif"
        ]

    @property
    def print_status(self) -> List[str]:
        return [
            "Printed", "Solid"
        ]

    @property
    def colors(self) -> List[str]:
        return [
            "Orange", "Red", "Green", "Grey", "Pink", 
            "Blue", "Purple", "White", "Black", "Yellow", "Beige",
            "Maroon", "Burgundy", "Brown"
        ]

    @property
    def materials(self) -> List[str]:
        return [
            "Cotton", "Linen", "Chiffon", "Satin", "Velvet", "Denim",
            "Lace", "Polyester", "Georgette", "Silk", "Sequin",
            "Crepe", "Jersey", "Knit", "Ribbed", "Modal", "Tulle",
            "Organza", "Leather", "Jacquard", "Tweed"
        ]

class ApparelAttributesMixin:
    """Mixin class that provides common attribute properties"""
    def __init__(self):
        self._common_attrs = CommonApparelAttributes()
        self._initialize_attributes()

    def _initialize_attributes(self):
        """Initialize apparel-specific attributes. Override in child classes."""
        self._lengths = []
        self._necklines = None
        self._sleeves = None
        self._fits = []
        self._design_features = []

    @property
    def print_types(self) -> List[str]:
        return self._common_attrs.print_types

    @property
    def print_color_styles(self) -> List[str]:
        return self._common_attrs.print_color_styles

    @property
    def print_sizes(self) -> List[str]:
        return self._common_attrs.print_sizes

    @property
    def print_status(self) -> List[str]:
        return self._common_attrs.print_status

    @property
    def colors(self) -> List[str]:
        return self._common_attrs.colors

    @property
    def materials(self) -> List[str]:
        return self._common_attrs.materials

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