# Generated by Django 5.1.1 on 2024-11-07 14:46

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('feed', '0008_product_image_embeddings_product_text_embeddings'),
    ]

    operations = [
        migrations.RenameField(
            model_name='product',
            old_name='product_description',
            new_name='description',
        ),
        migrations.RenameField(
            model_name='product',
            old_name='product_image_url',
            new_name='image_url',
        ),
        migrations.RenameField(
            model_name='product',
            old_name='product_name',
            new_name='name',
        ),
        migrations.RenameField(
            model_name='product',
            old_name='product_secondary_images',
            new_name='secondary_images',
        ),
    ]
