from django.db import models

class ImagePair(models.Model):
    title = models.CharField(max_length=50, blank=True)
    pre_image = models.ImageField(upload_to='pre_images/')
    post_image = models.ImageField(upload_to='post_images/')
    output_segment = models.ImageField(upload_to='segment_masks/', blank=True)
    output_damage = models.ImageField(upload_to='damage_masks/', blank=True)
    damage_percentage = models.FloatField(blank=True,null=True)