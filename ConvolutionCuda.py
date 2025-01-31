from numba import cuda
import numpy as np
from math import exp

@cuda.jit
def gaussian_gpu(sigma,kernel_size,kernel):
    m = kernel_size //2
    n = kernel_size //2

    x = cuda.threadIdx.x
    y = cuda.threadIdx.y

    kernel[x,y] = exp(-((x-m) ** 2 + (y-n) ** 2 ) / (2*sigma**2))

sigma = 5.3
kernel_size = 30

kernel = np.zeros((kernel_size, kernel_size), np.float32)
d_kernel = cuda.to_device(kernel)

@cuda.jit
def convolve(result, mask , image):
    i,j = cuda.grid(2)
    image_rows , image_cols = image_shape
    if(i >= image_rows) or (j >= image_cols):
        return
    
    delta_rows = mask.shape[0]
    delta_cols = mask.shape[1]

    s = 0 
    for k in range(mask.shape[0]):
        for l in range(mask.shape[1]):
            i_k = i - k + delta_rows
            j_l = j - l + delta_cols
            if (i_k >= 0 ) and (i_k < image_rows) and (j_l >= 0) and (j_l <= image_cols):
                s += mask[k,l] * image[i_k, j_l]
    result[i,j] = s

from PIL import Image , ImageOps

image = np.asarray(ImageOps.grayscale(Image.open("image.jpg")))

d_image = cuda.to_device(image)

d_result = cuda.device_array_like(image)

gaussian_gpu[(1,), (kernel_size,kernel_size)](sigma,kernel_size,d_kernel)

blockdim = (32,32)
griddim(image)