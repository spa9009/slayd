# Generated by Django 5.1.1 on 2024-11-19 18:42

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('account', '0002_remove_user_email_user_age_user_gender'),
        ('activity', '0004_alter_useractivity_unique_together_useractivity_post_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Follow',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('publisher_type', models.CharField(choices=[('brand', 'BRAND'), ('influencer', 'INFLUENCER')], max_length=10)),
                ('publisher', models.CharField(max_length=50)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='follow', to='account.user')),
            ],
            options={
                'constraints': [models.UniqueConstraint(fields=('user', 'publisher'), name='unique_user_publisher_action')],
            },
        ),
    ]