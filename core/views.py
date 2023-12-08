from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ImagePairForm
from .models import ImagePair
from .utils import process_images_with_model  
from PIL import Image

def upload_images(request):
    if request.method == 'POST':

        form = ImagePairForm(request.POST, request.FILES)

        if form.is_valid():

            pre_image = request.FILES['pre_image']
            post_image = request.FILES['post_image']
            
            # validate_uploaded_images(pre_image, post_image, request,form)

            image_pair = form.save()

            process_images_with_model(image_pair)

            return redirect('result', pk=image_pair.pk)
    else:
        form = ImagePairForm()
    return render(request, 'core/upload_images.html', {'form': form})


def list_results(request):
    recent_results = ImagePair.objects.all()
    return render(request, 'core/list_results.html', {'recent_results':recent_results})

                                                       
def show_result(request, pk):
    image_pair = ImagePair.objects.get(pk=pk)
    return render(request, 'core/result.html', {'image_pair': image_pair})

def edit_result(request, pk):

    def edit_image_pair(request, pk):
        recent_results = ImagePair.objects.all()

    image_pair = get_object_or_404(ImagePair, pk=pk)

    if request.method == 'POST':
        form = ImagePairForm(request.POST, request.FILES, instance=image_pair)

        if form.is_valid():
            pre_image = request.FILES['pre_image']
            post_image = request.FILES['post_image']
            validate_uploaded_images(pre_image, post_image, request)
            image_pair = form.save()
            process_images_with_model(image_pair)
            return redirect('result', pk=image_pair.pk)
    else:
        image_pair = ImagePair.objects.get(pk=pk)
        return render(request, 'core/edit.html', {'image_pair': image_pair})

def delete_result(request, pk):
    image_pair = ImagePair.objects.get(pk=pk)
    if request.method == 'POST':
        image_pair.delete()
    return redirect('list_results')


def validate_uploaded_images(pre_image, post_image, request, form):
    def get_image_dimensions(image):
        with Image.open(image) as img:
            return img.size

    min_size = 256

    if get_image_dimensions(pre_image) != get_image_dimensions(post_image):
        return render(request, 'core/upload_images.html', {'form': form})
    if any(dim <= min_size for dim in get_image_dimensions(pre_image)):
        return render(request, 'core/upload_images.html', {'form': form})
    if any(dim <= min_size for dim in get_image_dimensions(post_image)):
        return render(request, 'core/upload_images.html', {'form': form})


