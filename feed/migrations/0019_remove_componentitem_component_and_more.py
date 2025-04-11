# Generated by Django 5.1.1 on 2025-04-11 14:30

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('feed', '0018_myntraproducttags'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='componentitem',
            name='component',
        ),
        migrations.RemoveField(
            model_name='componentitem',
            name='item',
        ),
        migrations.RemoveField(
            model_name='curation',
            name='curation_type',
        ),
        migrations.RemoveField(
            model_name='curation',
            name='parent_curation',
        ),
        migrations.AddField(
            model_name='curation',
            name='created_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AddField(
            model_name='curation',
            name='products',
            field=models.ManyToManyField(related_name='curations', to='feed.myntraproducts'),
        ),
        migrations.AddField(
            model_name='curation',
            name='title',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.DeleteModel(
            name='Component',
        ),
        migrations.DeleteModel(
            name='ComponentItem',
        ),
        migrations.DeleteModel(
            name='Item',
        ),
    ]
