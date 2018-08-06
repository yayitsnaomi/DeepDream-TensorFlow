import numpy as np
import tensorflow as tf
import PIL.Image
import random
import math
import re
import base64
import sys
import json
import os
import warnings
import backtrace
import time

from io import BytesIO, StringIO
import inception5h as inception5h
from scipy.ndimage.filters import gaussian_filter

backtrace.hook(
    reverse=False,
    align=False,
    strip_path=False,
    enable_on_envvar_only=False,
    on_tty=False,
    conservative=False,
    styles={})

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
warnings.filterwarnings('ignore', '.*do not.*',)

model = inception5h.Inception5h()
session = tf.InteractiveSession(graph=model.graph)

""" SYS ARGVS """

imagePath   = str(sys.argv[1])
tensorLayer = int(sys.argv[2])
tensorModel = int(sys.argv[3])


def normalize_image(x):
    x_min = x.min()
    x_max = x.max()

    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def resize_image(image, size=None, factor=None):
    if factor is not None:
        size = np.array(image.shape[0:2]) * factor
        size = size.astype(int)
    else:
        size = size[0:2]

    size = tuple(reversed(size))
    img = np.clip(image, 0.0, 255.0)
    img = img.astype(np.uint8)
    img = PIL.Image.fromarray(img)

    img_resized = img.resize(size, PIL.Image.LANCZOS)

    img_resized = np.float32(img_resized)

    return img_resized



def get_tile_size(num_pixels, tile_size=400):

    num_tiles = int(round(num_pixels / tile_size))
    num_tiles = max(1, num_tiles)
    actual_tile_size = math.ceil(num_pixels / num_tiles)

    return actual_tile_size


def tiled_gradient(gradient, image, tile_size=400):
    grad = np.zeros_like(image)
    x_max, y_max, _ = image.shape
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    x_tile_size4 = x_tile_size // 4
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    y_tile_size4 = y_tile_size // 4

    x_start = random.randint(-3 * x_tile_size4, -x_tile_size4)

    while x_start < x_max:
        x_end = x_start + x_tile_size
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)
        y_start = random.randint(-3 * y_tile_size4, -y_tile_size4)

        while y_start < y_max:
            y_end = y_start + y_tile_size
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)
            img_tile = image[x_start_lim:x_end_lim,
                       y_start_lim:y_end_lim, :]
            feed_dict = model.create_feed_dict(image=img_tile)
            g = session.run(gradient, feed_dict=feed_dict)
            g /= (np.std(g) + 1e-8)
            grad[x_start_lim:x_end_lim,
            y_start_lim:y_end_lim, :] = g
            y_start = y_end

        x_start = x_end

    return grad


def optimize_image(layer_tensor, image,
                   num_iterations=10, step_size=3.0, tile_size=400,
                   show_gradient=True):

    img = image.copy()
    gradient = model.get_gradient(layer_tensor)
    for i in range(num_iterations):

        logger({'task': 'optimizer_single_iteration', 'log': i})

        grad = tiled_gradient(gradient=gradient, image=img)
        sigma = (i * 4.0) / num_iterations + 2.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma * 2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma * 0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)
        step_size_scaled = step_size / (np.std(grad) + 1e-8)
        img += grad * step_size_scaled

    return img


def recursive_optimize(layer_tensor, image,
                       num_repeats=4, rescale_factor=0.7, blend=0.2,
                       num_iterations=9, step_size=3.0,
                       tile_size=400):

    if num_repeats > 0:

        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))
        img_downscaled = resize_image(image=img_blur,
                                      factor=rescale_factor)
        img_result = recursive_optimize(layer_tensor=layer_tensor,
                                        image=img_downscaled,
                                        num_repeats=num_repeats - 1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size,
                                        tile_size=tile_size)

        img_upscaled = resize_image(image=img_result, size=image.shape)
        image = blend * image + (1.0 - blend) * img_upscaled


    logger({'task': 'recusrive_optimizer', 'log': num_repeats})

    img_result = optimize_image(layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size,
                                tile_size=tile_size)

    return img_result


def load_image(imgPath):
    image = PIL.Image.open(imgPath)
    return np.float32(image)


def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')

def generate_deep_dream(imgPath):
    loaded_img = load_image(imgPath)
    layer_tensor = model.layer_tensors[tensorLayer][:, :, :, tensorModel]

    img_result = recursive_optimize(layer_tensor=layer_tensor, image=loaded_img,
                     num_iterations=25, step_size=5.5, rescale_factor=0.7,
                     num_repeats=10, blend=0.5, tile_size=1000)

    ts = int(time.time())
    generated_filename = './generated/' + str(ts) + '-generated.jpg'
    os.rename(imagePath, './generated/' + str(ts) + '-original.jpg')
    save_image(img_result, generated_filename)


def logger(log):
    print(json.dumps(log))
    sys.stdout.flush()


def performance(millis):
    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis/(1000*60*60))%24

    return ("%d:%d:%d" % (hours, minutes, seconds))

"""
INITIALIZE PROCESS
"""

logger({'task': 'start', 'log': 'Starting deepdream'})

start_time = time.time()
""" GENERATE DEEPDREAM """
generate_deep_dream(imagePath)

millis = (time.time() - start_time)

""" SEND CALLBACK """
logger({'task': 'exit', 'performance': performance(millis)})