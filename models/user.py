from django.db import models

class User(models.Model): 
    username = models.Charfield(max_length=20, unique=True)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=10, unique=True)
    password = models.BooleanField(default=False)
    is_verified = models.BooleanField(default=False)
    signup_timestamp = models.TimeField


    def __str__(self):
        return self.phone