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

