# Generated by Django 4.2.7 on 2023-11-26 18:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_imagepair_output_damage_imagepair_output_segment'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagepair',
            name='title',
            field=models.CharField(blank=True, max_length=50),
        ),
        migrations.AlterField(
            model_name='imagepair',
            name='output_damage',
            field=models.ImageField(blank=True, upload_to='damage_masks/'),
        ),
        migrations.AlterField(
            model_name='imagepair',
            name='output_segment',
            field=models.ImageField(blank=True, upload_to='segment_masks/'),
        ),
    ]
