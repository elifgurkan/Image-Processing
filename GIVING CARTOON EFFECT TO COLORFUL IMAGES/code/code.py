# -*- coding: utf-8 -*-
"""Assignment1_415.ipynb

Original file is located at
    https://colab.research.google.com/drive/1zosXdLL_znVeMKuDCtC1Qqb3sei-3owy
"""

from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
from scipy.ndimage import median_filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('image16.jpg')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

"""çok aydınlık fotoğraflardan ziyade renkli, daha az aydınlık (ama karanlık olmayan) fotoğraflar daha iyi"""

sigma = 2
smoothed_image = ndimage.gaussian_filter(img, sigma=(sigma, sigma, 0))
plt.imshow(cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB))
plt.title('Smoothed Image (Gaussian Filter)')

plt.tight_layout()
plt.show()

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
L, a, b = cv2.split(lab_image)
custom_kernel = np.ones((5, 5)) / 25  # 5x5 box filter

# applying convolution with the custom kernel to the L channel
smoothed_L = convolve(L, custom_kernel)
smoothed_L = np.clip(smoothed_L, 0, 255).astype(np.uint8)

# new Lab image with the smoothed L channel
smoothed_lab_image = cv2.merge((smoothed_L, a, b))
smoothed_rgb_image = cv2.cvtColor(smoothed_lab_image, cv2.COLOR_Lab2BGR)

plt.figure(figsize=(10,5))
plt.imshow(cv2.cvtColor(smoothed_rgb_image, cv2.COLOR_BGR2RGB))
plt.title('Smoothed Image (Custom Convolution)')
plt.tight_layout()
plt.show()

plt.imshow(cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB))

# standard deviations for the Gaussian filters
sigma1 = 1.5
sigma2 = 2

gaussian1 = gaussian_filter(gray_image, sigma=sigma1)
gaussian2 = gaussian_filter(gray_image, sigma=sigma2)

# DoG
edges = gaussian1 - gaussian2

threshold_value = 7
edges[edges < threshold_value] = 0
edges[edges >= threshold_value] = 255

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge-Detected Image (DoG)')

plt.tight_layout()
plt.show()

lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
L, a, b = cv2.split(lab)

# the number of quantization levels
quantization_level = 2

# quantize the L channel(The formula used scales the values to the desired number of levels)
quantized_L = np.round(L * (quantization_level-1) / 100) * (100 / (quantization_level-1))

# Merging the quantized L channel with the original a and b channels
#The .astype('uint8') ensures that the data type is converted to 8-bit unsigned integer, which is the expected type for the Lab color space.
quantized_lab_image = cv2.merge((quantized_L.astype('uint8'), a, b))
quantized_rgb_image = cv2.cvtColor(quantized_lab_image, cv2.COLOR_Lab2BGR)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(quantized_rgb_image, cv2.COLOR_BGR2RGB))
plt.title('Quantized Image')

plt.tight_layout()
plt.show()

normalized_edges = edges / 255.0

weight_quantized_image = 0.9  # adjust to balance the contribution
weight_edges = 1 - weight_quantized_image

combined_image = (weight_quantized_image * quantized_rgb_image) + (weight_edges * normalized_edges[..., np.newaxis])
combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.title('Combined Image')

plt.tight_layout()
plt.show()