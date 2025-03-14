# Generated by Django 5.1.1 on 2024-10-16 17:13

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('feed', '0003_remove_product_product_id_post_media_taggedproduct'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='post',
            name='updated_at',
        ),
        migrations.AddField(
            model_name='post',
            name='description',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='post',
            name='title',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name='post',
            name='created_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='taggedproduct',
            name='post',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='tagged_products', to='feed.post'),
        ),
    ]
