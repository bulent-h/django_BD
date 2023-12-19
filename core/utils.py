import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
import numpy as np
from PIL import Image 
from django.core.files.uploadedfile import InMemoryUploadedFile
from imageio.v2 import imread
from .model import  Conv2DBNLayer, MultiResBlock,AttentionBlock
from patchify import patchify, unpatchify
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO



def process_images_with_model(image_pair):

    patch_size = 256
    scaler = MinMaxScaler()

    model = keras.models.load_model('model.keras', 
                                    custom_objects={"Conv2DBNLayer":Conv2DBNLayer ,"MultiResBlock": MultiResBlock,"AttentionBlock":AttentionBlock })
   

    pre_img = tf.io.decode_png(tf.io.read_file(image_pair.pre_image.path), channels=3)
    post_img = tf.io.decode_png(tf.io.read_file(image_pair.post_image.path), channels=3)

    org_shape = pre_img.shape

    pre_img = expand_image_to_multiple_of_256(pre_img)
    post_img = expand_image_to_multiple_of_256(post_img)

    SIZE_X = (pre_img.shape[1]//patch_size)*patch_size 
    SIZE_Y = (pre_img.shape[0]//patch_size)*patch_size 

  
    large_pre_img = Image.fromarray(pre_img)
    large_post_img = Image.fromarray(post_img)

    large_pre_img = large_pre_img.crop((0 ,0, SIZE_X, SIZE_Y))  
    large_post_img = large_post_img.crop((0 ,0, SIZE_X, SIZE_Y))  

    large_pre_img = np.array(large_pre_img)
    large_post_img = np.array(large_post_img)

    patches_pre_img = patchify(large_pre_img, (patch_size, patch_size, 3), step=patch_size)
    patches_post_img = patchify(large_post_img, (patch_size, patch_size, 3), step=patch_size)

    patches_pre_img = patches_pre_img[:,:,0,:,:,:]
    patches_post_img = patches_post_img[:,:,0,:,:,:]
    patched_pred_segment = []
    patched_pred_damage = []

    for i in range(patches_pre_img.shape[0]):
        for j in range(patches_pre_img.shape[1]):

            single_patch_pre_img = patches_pre_img[i,j,:,:,:]
            single_patch_post_img = patches_post_img[i,j,:,:,:]

            single_patch_pre_img = scaler.fit_transform(single_patch_pre_img.reshape(-1, single_patch_pre_img.shape[-1])).reshape(single_patch_pre_img.shape)
            single_patch_pre_img = np.expand_dims(single_patch_pre_img, axis=0)

            single_patch_post_img = scaler.fit_transform(single_patch_post_img.reshape(-1, single_patch_post_img.shape[-1])).reshape(single_patch_post_img.shape)
            single_patch_post_img = np.expand_dims(single_patch_post_img, axis=0)

            pred_post_damage, pred_pre_segment = model.predict([single_patch_pre_img, single_patch_post_img])

            damage_pred = np.argmax(pred_post_damage, axis=-1)
            segment_pred = np.argmax(pred_pre_segment, axis=-1)


            patched_pred_damage.append(damage_pred)
            patched_pred_segment.append(segment_pred)

    patched_pred_damage = np.array(patched_pred_damage)
    patched_pred_segment = np.array(patched_pred_segment)

    patched_pred_damage = np.squeeze(patched_pred_damage)
    patched_pred_segment = np.squeeze(patched_pred_segment)

    patched_pred_damage = np.reshape(patched_pred_damage, [patches_pre_img.shape[0], patches_pre_img.shape[1],
                                                patches_pre_img.shape[2], patches_pre_img.shape[3]])

    patched_pred_segment = np.reshape(patched_pred_segment, [patches_pre_img.shape[0], patches_pre_img.shape[1],
                                                patches_pre_img.shape[2], patches_pre_img.shape[3]])

    damage_image = unpatchify(patched_pred_damage, (large_post_img.shape[0], large_post_img.shape[1]))
    segment_image = unpatchify(patched_pred_segment, (large_pre_img.shape[0], large_pre_img.shape[1]))
    
    damage_image = remove_padding(damage_image,org_shape)
    segment_image = remove_padding(segment_image,org_shape)
    
    image_pair.damage_percentage = calculate_damage(damage_image)

    damage_image = convert_to_png(damage_image)
    segment_image = convert_to_png(segment_image)


    image_pair.output_damage.save('damage_image.jpg', damage_image)
    image_pair.output_segment.save('segment_image.jpg', segment_image)


def calculate_damage(np_image):
    count_1 = np.count_nonzero(np_image == 1)
    count_2 = np.count_nonzero(np_image == 2)
    result = (count_2/(count_1+count_2))*100
    result = round(result, 3)
    return result


def preprocess_image(image_path):
    img = imread(image_path)
    img = np.nan_to_num(img, 0)
    img = tf.convert_to_tensor(img,dtype=tf.float32)/255 
    return img

def convert_to_png(class_array):
    color_map = {
        0: (0, 0, 0),
        1: (255, 255, 255),
        2: (255, 0, 0)
    }
    height, width = class_array.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_label, color in color_map.items():
        rgb_image[class_array == class_label] = color
    image = Image.fromarray(rgb_image, 'RGB')

    image_io = BytesIO()
    image.save(image_io, format='PNG')  
    image_io.seek(0)

    return image_io

def expand_image_to_multiple_of_256(np_image):
    
    height, width, channels = np_image.shape

    pad_height = (256 - height % 256) % 256
    pad_width = (256 - width % 256) % 256

    padded_image = np.zeros((height + pad_height, width + pad_width, channels), dtype=np.uint8)

    padded_image[:height, :width, :] = np_image

    return padded_image

def remove_padding(image, original_shape):
    height, width, channels = original_shape

    original_image = image[:height, :width]

    return original_image

