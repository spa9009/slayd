import re
from collections import Counter
from django.db import transaction
from feed.models import Product
from django.core.management.base import BaseCommand

GENDERS = {
    "men": "Men",
    "women": "Women",
    "unisex": "Unisex",
    "kids": "Kids"
}

CATEGORIES = {
    "t-shirt": "T-Shirts",
    "sweatshirt": "Hoodies and Sweatshirts",
    "tshirt": "T-Shirts",
    "pant": "Jeans and Trousers",
    "trouser": "Jeans and Trousers",
    "cargo": "Jeans and Trousers",
    "jeans": "Jeans and Trousers",
    "jogger": "Jeans and Trousers",
    "tee": "Tops",
    "top": "Tops",
    "hoodie": "Hoodies and Sweatshirts",
    "sweatshirt": "Hoodies and Sweatshirts",
    "shirt": "Shirts",
    "tank": "Tops",
    "polo": "T-Shirts",
    "short": "Shorts",
    "dress": "Dresses",
    "jacket": "Jackets",
    "shacket": "Jackets",
    "sweater": "Hoodies and Sweatshirts",
    "turtleneck": "Hoodies and Sweatshirts",
    "cardigans": "Hoodies and Sweatshirts",
    "skort": "Skirts",
    "skirt": "Skirts",
    "bootcut": "Jeans and Trousers",
    "jumpsuit": "Jumpsuit and Dungaree",
    "dungaree": "Jumpsuit and Dungaree",
    "leggings": "Jeans and Trousers",
    "vest": "T-Shirts",
    "corset": "Tops",
    "jersey": "Tshirts",
    "bodysuit": "Bodysuits",
    "cami": "Tops",
    "blazer": "Blazers and Tracksuits",
    "tracksuit": "Blazers and Tracksuits",
    "denim": "Jeans and Trousers",
    "co ord": "Coords",
    "bralette": "Tops",
    "set": "Coords",
    "leg": "Jeans and Trousers",
    "gilet": "Jackets",
    "zipper": "Hoodies and Sweatshirts",
    "bodycon": "Dresses",
    "full sleeves": "Hoodies and Sweatshirts",
    "cap": "Caps",
    "glasses": "Glasses"
}

SUBCATGORIES = {
    "T-Shirts" : {
        "oversized": "Oversized T-Shirts",
        "polo": "Polo T-Shirts",
        "vest": "Vests",
        "heavyweight": "Heavyweight T-Shirts",
        "printed": "Printed T-Shirts",
        "relaxed": "Relaxed Fit T-Shirts",
        "fitted": "Fit T-Shirts",
        "polyamide": "Polyamide T-Shirts",
        "full sleeves": "Full Sleeves",
        "knitted": "Knitted T-Shirts",
        "crochet": "Crochet",
        "t-shirts": "T-Shirts"

    },
    "Hoodies and Sweatshirts": {
        "hoodies": "Hoodies",
        "sweatshirt": "Sweatshirts",
        "zipper": "Zipper"
    },
    "Jeans and Trousers": {
        "corduroy": "Corduroy",
        "straight": "Straight Fit",
        "parachute": "Parachute",
        "loose": "Loose Fit",
        "sweatpant": "Sweatpants",
        "korean": "Korean Pants",
        "cargo": "Cargos",
        "bootcut": "Bootcut",
        "bell": "Bell Bottom",
        "utility": "Utility",
        "wide": "Wide-Leg",
        "track": "Track Pants",
        "baggy": "Baggy Pants",
        "fold" : "Fold-Over",
        "stretchable": "Stretchable",
        "low waist": "Low-Waist",
        "sporty": "Sweatpants",
        "boot-cut": "Bootcut",
        "flare": "Bootcut",
        "skinny": "Slim fit",
        "slim": "Slim fit",
        "slit": "Slit",
        "regular": "Regular",
        "mom": "Mom Jeans",
        "ripped": "Ripped",
        "hugging": "Slim fit",
        "cut-out": "Ripped",
        "trouser": "Trousers",
        "pant" : "Pants",
        "jeans": "Jeans",
        "leather": "Leather",
        "denim": "Denims",
        "leggings": "leggings",
        "jogger": "Sweatpants"
    },
    "Tops": {
        "polyamide": "Polyamide",
        "crop": "Crop Top",
        "tube": "Tube Top",
        "tank": "Tank Top",
        "corset": "Corset tops",
        "denim": "Denim Tops",
        "cami": "Cami Tops",
        "square": "Square Neck",
        "top": "Tops"
    },  
    "Blazers and Tracksuits": {
        "blazers": "Blazers",
        "track": "Tracksuits"
    },
    "Shirts": {
        "oversized": "Oversized",
        "regular": "Regular Fit",
        "corduroy": "Corduroy",
        "print": "Printed",
        "mesh": "Mesh",
        "resort": "Resort",
        "linen": "Linen",
        "crochet": "Crochet",
        "cuban": "Cuban",
        "bowling": "Bowling",
        "overshirt": "Overshirt",
        "knitted": "Knitted"
    },
    "Jackets": {
        "denim": "Denim Jacket",
        "gilet": "Gilet",
        "shacket": "Shacket",
        "jackets": "Jackets"
    },
    "Jumpsuit and Dungaree": {
        "jumpsuit": "Jumpsuit",
        "dungaree": "Dungaree"
    },
    "Dresses": {
        "sleeveless": "Sleeveless Dress",
        "cami": "Cami Dress",
        "short": "Short Dress",
        "midi": "Midi Dress",
        "maxi": "Maxi Dress",
        "knitted": "Knitted Dress",
        "bodycon": "Bodycon Dress",
        "backless": "Backless Dress",
        "shirt": "Shirt Dress",
        "ribbed": "Ribbed Dress",
        "tube": "Tube Dress",
        "blazer": "Blazer Dress",
        "dress": "Dress"
    },
    "Coords": {
        "coords": "Co Ord Sets",
        "sets": "Co Ord Sets"
    }, 
    "Skirts": {
        "mini": "Short Skirts",
        "pleated": "Pleated Skirts",
        "short": "Short Skirts",
        "midi": "Midi Skirts",
        "pencil": "Pencil Skirts",
        "denim": "Denim Skirts",
        "wrap": "Wrap Skirts",
        "skirt": "Skirts",
        "skort": "Skorts"
    },
    "Shorts": {
        "cycling": "Cycling Shorts",
        "bermudas": "Bermudas",
        "denim": "Denim Shorts",
        "jogger": "Jogger Shorts",
        "jeans": "Jeans Shorts",
        "chino": "Chino Shorts",
        "cargo": "Cargo Shorts",
        "casual": "Casuals",
        "shorts": "Shorts"
    },
    "Bodysuits": {
        "seamless": "Seamless Bodysuits",
        "full sleeve": "Full Sleeves Bodysuits",
        "polyamide": "Polyamide Bodysuits",
        "highneck": "Highneck Bodysuits",
        "bodysuit": "Bodysuits"
    },
    "Caps": {
        "caps": "Caps"
    },
    "Glasses": {
        "glasses": "Glasses"
    }
}

GENDER_CATEGORY_MAPPING = {
    "Female" : ["Tops", "Dresses", "Skirts", "Bodysuits"]
}

def extract_gender(name, category, subcategory, extracted_category) :

    ## Overriding gender in case extracted category is present in GENDER_CATEGORY_MAPPING 
    for gender, category_map in GENDER_CATEGORY_MAPPING.items(): 
        for cat in category_map: 
            if cat in extracted_category: 
                return gender
            
    text = f"{name} {category} {subcategory}".lower()

    for keyword, gender in GENDERS.items():
        if keyword in text: 
            return gender
    return "Unisex"


def extract_category(name, category) :
    text = f"{name} {category}".lower()

    for keyword, _category in CATEGORIES.items():
        if keyword in text: 
            return _category
    return category


def extract_subcategory(name, category, subcategory, extracted_category):
    text = f"{name} {category} {subcategory}".lower()

    for cat, map in SUBCATGORIES.items():
        if cat == extracted_category :
            for key, subcat in map.items():
                if key in text: 
                    return subcat
    return extracted_category


class Command(BaseCommand):
    help = 'Extract embeddings for products and save them to the database'

    def handle(self, *args, **kwargs):
        products = Product.objects.all()

        category_counter = Counter()
        subcategory_counter = Counter()

        results = []

        for product in products:
            name = product.name
            category = product.category
            subcategory = product.subcategory

            category_counter[category] += 1
            subcategory_counter[subcategory] += 1

            new_category = extract_category(name, category)

            gender = extract_gender(name, category, subcategory, new_category)

            new_subcategory = extract_subcategory(name, category, subcategory, new_category)

            print(f"ID: {product.id}. Gender, category and subcategory are: {gender}, {new_category}, {new_subcategory}")
            results.append({
                "product_id": product.id,
                "name": name,
                "category": new_category,
                "subcategory": new_subcategory,
                "gender": gender,
            })

        print("Most Common Categories:", category_counter.most_common(5))
        print("Most Common Subcategories:", subcategory_counter.most_common(5))
        with transaction.atomic():
            for result in results: 
                product = Product.objects.get(id=result["product_id"])
                product.gender = result["gender"]
                product.category = result["category"]
                product.subcategory = result["subcategory"]
                product.save()

