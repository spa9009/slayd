# Generated by Django 5.1.1 on 2024-12-30 08:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('feed', '0011_remove_product_image_embedding_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='product',
            name='gender',
            field=models.CharField(default='Unisex', max_length=10),
            preserve_default=False,
        ),
    ]