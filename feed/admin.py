from django.contrib import admin
from models.product import Product
from models.user import User

admin.site.register(Product)
admin.site.register(User)