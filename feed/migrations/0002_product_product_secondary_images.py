# Generated by Django 5.1.1 on 2024-09-30 18:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('feed', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='product',
            name='product_secondary_images',
            field=models.JSONField(default=[]),
            preserve_default=False,
        ),
    ]
