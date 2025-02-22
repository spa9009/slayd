from django.core.management.base import BaseCommand
from django.db import transaction
from feed.models import Curation, Component, Item, ComponentItem
import pandas as pd

data = {
  "curations": [
    {
      "title": "Inspiration 1",
      "image": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/22ba8b5cabce082bbedc9e041f66f548+1%402x.png",
      "categories": [
        {
          "name": "Jump Suits",
          "products": [
            {
              "name": "Trendyol Women Blue Basic Pure Cotton Jumpsuit with Belt",
              "mrp": 3899,
              "price": 3314,
              "marketplace": "Myntra",
              "redirect_link": "https://www.myntra.com/jumpsuit/trendyol/trendyol-women-blue-basic-pure-cotton-jumpsuit-with-belt-/17813704/buy",
              "s3_url": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/Screenshot+2025-02-22+at+2.27.52%E2%80%AFPM.png"
            },
            {
              "name": "Vero Moda Women Solid Shirt Collar Culotte Jumpsuit",
              "mrp": 6999,
              "price": 3499,
              "marketplace": "Myntra",
              "redirect_link": "https://www.myntra.com/jumpsuit/veromoda/vero-moda-women-solid-shirt-collar-culotte-jumpsuit/30915418/buy",
              "s3_url": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/Screenshot+2025-02-22+at+2.28.07%E2%80%AFPM.png"
            },
            {
              "name": "20Dresses Pure Cotton Long Sleeves Basic Jumpsuit",
              "mrp": 3595,
              "price": 1258,
              "marketplace": "Myntra",
              "redirect_link": "https://www.myntra.com/jumpsuit/20dresses/20dresses-pure-cotton-long-sleeves-basic-jumpsuit/21648152/buy",
              "s3_url": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/Screenshot+2025-02-22+at+2.28.33%E2%80%AFPM.png"
            },
            {
              "name": "H&M Denim Boilersuit",
              "mrp": 5999,
              "price": 3899,
              "marketplace": "Myntra",
              "redirect_link": "https://www.myntra.com/hatke/a/products/30016990/buy?ep=rd_111108760",
              "s3_url": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/Screenshot+2025-02-22+at+2.28.50%E2%80%AFPM.png"
            },
            {
              "name": "StyleStone Basic Jumpsuit",
              "mrp": 2399,
              "price": 1199,
              "marketplace": "Myntra",
              "redirect_link": "https://www.myntra.com/jumpsuit/stylestone/stylestone-basic-jumpsuit/30131825/buy",
              "s3_url": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/Screenshot+2025-02-22+at+2.29.05%E2%80%AFPM.png"
            },
            {
              "name": "ONLY Denim Jumpsuit with Zip Closure",
              "mrp": 3699,
              "price": 1295,
              "marketplace": "Ajio",
              "redirect_link": "https://www.ajio.com/only-denim-jumpsuit-with-zip-closure/p/440984123_blue",
              "s3_url": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/Screenshot+2025-02-22+at+2.29.49%E2%80%AFPM.png"
            }
          ]
        }
      ]
    },
    {
      "title": "Inspiration 2",
      "image": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/44ce76e628575d2f91c8c3e3f5f1e3cd+(1).jpg",
      "categories": [
        {
          "name": "Jackets",
          "products": [
            {
              "name": "ONLY Women Spread Collar Solid Cotton Casual Denim Jacket",
              "mrp": 3499,
              "price": 1749,
              "marketplace": "Myntra",
              "redirect_link": "https://www.myntra.com/hatke/a/products/31183225/buy?ep=rd_111109108",
              "s3_url": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/Screenshot+2025-02-22+at+3.02.15%E2%80%AFPM.png"
            },
            {
              "name": "Saman Fashion Wear Solid 3/4th Sleeve Denim Jacket",
              "mrp": 272,
              "marketplace": "Meesho",
              "redirect_link": "https://www.meesho.com/saman-fashion-wear-solid-34th-sleeve-denim-jacket-for-women-womens-stylish-jacket-trendy-casual-wear-jackets-for-women-attractive-denim-fabulous-western-pretty-fashionable-denim-jacketsdark-blue/p/2ixfer",
              "s3_url": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/Screenshot+2025-02-22+at+3.02.30%E2%80%AFPM.png"
            }
          ]
        }
      ]
    },
    {
      "title": "Inspiration 3",
      "image": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/61f30df9bfb62c57e278cded37c8ffa9+(1).jpg",
      "categories": [
        {
          "name": "Tops",
          "products": [
            {
              "name": "Fame Forever by Lifestyle Women Collarless Washed Solid Cotton Casual Denim Jacket",
              "mrp": 1299,
              "price": 1299,
              "marketplace": "Myntra",
              "redirect_link": "https://www.myntra.com/jackets/fame+forever+by+lifestyle/fame-forever-by-lifestyle-women-collarless-washed-solid-cotton-casual-denim-jacket/32281664/buy",
              "s3_url": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/Screenshot+2025-02-22+at+3.57.33%E2%80%AFPM.png"
            },
            {
              "name": "StyleCast Denim Top",
              "mrp": 3049,
              "price": 1494,
              "marketplace": "Myntra",
              "redirect_link": "https://www.myntra.com/tops/stylecast/stylecast-denim-top/30731645/buy",
              "s3_url": "https://feed-images-01.s3.ap-south-1.amazonaws.com/media/Screenshot+2025-02-22+at+4.00.03%E2%80%AFPM.png"
            }
          ]
        }
      ]
    }
  ]
}



class Command(BaseCommand):
    help = 'Upload curations to the database'

    def handle(self, *args, **kwargs):
        with transaction.atomic():
            curation = Curation.objects.create(
                curation_type='MULTI_INSPIRATION'
            )

            for curation_data in data["curations"]:
                curation_single = Curation.objects.create(
                    curation_type='SINGLE',
                    curation_image=curation_data["image"],
                    parent_curation=curation
                )

                for category_data in curation_data["categories"]:
                    component = Component.objects.create(
                        curation=curation_single,
                        name=category_data["name"]
                    )

                    for product_data in category_data["products"]:
                        item = Item.objects.create(
                            name=product_data["name"],
                            price=product_data["mrp"],
                            discount_price=product_data.get("price", None),
                            marketplace=product_data.get("marketplace", None),
                            link=product_data["redirect_link"],
                            image_url=product_data["s3_url"]
                        )

                        ComponentItem.objects.create(
                            component=component,
                            item=item
                        )


                
                