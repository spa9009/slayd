# Generated by Django 5.1.1 on 2025-02-22 09:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('feed', '0013_item_curation_component_componentitem'),
    ]

    operations = [
        migrations.AddField(
            model_name='item',
            name='marketplace',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
