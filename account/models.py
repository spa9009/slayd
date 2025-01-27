from django.db import models


# Create your models here.
class User(models.Model):

    GENDERS = [
        ('MALE', 'male'),
        ('FEMALE', 'female'),
        ('NON_BINARY', 'non_binary')
    ]

    username = models.CharField(max_length=20, unique=True)
    password = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(choices=GENDERS)
    phone = models.CharField(max_length=10, unique=True)

    def __str__(self):
        return self.username
    

    # Create your models here.
class UserRecord(models.Model):

    GENDERS = [
        ('MALE', 'male'),
        ('FEMALE', 'female'),
        ('NON_BINARY', 'non_binary')
    ]

    username = models.CharField(max_length=20, unique=True)
    password = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(choices=GENDERS)
    phone = models.CharField(max_length=10, unique=True)

    def __str__(self):
        return self.username


class Brand:
    name = models.CharField(max_length=50)

    def __str__(self): 
        return self.brand
    
class UserPreferences(models.Model):
    AESTHETICS = [
        ('Soft Girl Core'),
        ('Urban and Edgy'),
        ('Creative, unique and experimental styles'),
        ('Sporty and Functional'),
        ('Nightlife and Glam')
    ]

    STYLES = [
        ('Off-Shoulder'),
        ('Deep neck'),
        ('Backless'),
        ('Visible tummy'),
        ('Short'),
        ('Tight fitting')
    ]   

    user = models.OneToOneField(UserRecord, on_delete=models.CASCADE, related_name='profile')
    aesthetics = models.JSONField(default=list)
    avoid_styles = models.JSONField(default=list)

    def __str__(self):
        return self.user