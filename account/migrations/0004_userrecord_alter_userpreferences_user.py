# Generated by Django 5.1.1 on 2025-01-27 16:13

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('account', '0003_userpreferences'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=20, unique=True)),
                ('password', models.CharField(max_length=100)),
                ('age', models.IntegerField()),
                ('gender', models.CharField(choices=[('MALE', 'male'), ('FEMALE', 'female'), ('NON_BINARY', 'non_binary')])),
                ('phone', models.CharField(max_length=10, unique=True)),
            ],
        ),
        migrations.AlterField(
            model_name='userpreferences',
            name='user',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='profile', to='account.userrecord'),
        ),
    ]
