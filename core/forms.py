from django import forms
from .models import ImagePair
from PIL import Image
from django.core.exceptions import ValidationError


class ImagePairForm(forms.ModelForm):
    
    class Meta:
        model = ImagePair
        fields = ['title', 'pre_image', 'post_image']

    
