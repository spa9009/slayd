# Generated by Django 5.1.7 on 2025-03-14 14:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('account', '0004_userrecord_alter_userpreferences_user'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='gender',
            field=models.CharField(max_length=10),
        ),
        migrations.AlterField(
            model_name='userrecord',
            name='gender',
            field=models.CharField(max_length=10),
        ),
    ]
