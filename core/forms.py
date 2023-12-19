from django import forms
from .models import ImagePair
from PIL import Image
from django.core.exceptions import ValidationError


class ImagePairForm(forms.ModelForm):
    
    class Meta:
        model = ImagePair
        fields = ['title', 'pre_image', 'post_image']

    def clean(self):
        cleaned_data = super().clean()
        pre_image = cleaned_data.get('pre_image')
        post_image = cleaned_data.get('post_image')

        if pre_image and post_image:
            pre_image_size = Image.open(pre_image).size
            post_image_size = Image.open(post_image).size

            if pre_image_size != post_image_size:
                raise forms.ValidationError("Both images must be equal in size.")

            if pre_image_size[0] < 256 or pre_image_size[1] < 256:
                raise forms.ValidationError("Image dimensions must be larger than 256x256.")

        return cleaned_data
    
