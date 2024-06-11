# -*- coding: utf-8 -*-
"""Assignment3_BBM415.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G6svqN337_dSsk7IB5KUIHelTcXR_ZVs
"""

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image1.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)

"""# Pixel-Level Features"""

# normalizing the features to have the same value interval
normalized_rgb = img / 255.0
height, width, _ = img.shape

# Normalizing spatial coordinates by dividing by the max width and height
normalized_x = np.tile(np.arange(width) / width, (height, 1))
normalized_y = np.tile(np.arange(height) / height, (width, 1)).T

pixel_features_rgb = normalized_rgb.reshape(-1, 3)  # Reshape to (num_pixels, 3)

# Feature for (b): RGB color and spatial location feature [R G B x y]
pixel_features_rgb_xy = np.zeros((height * width, 5))
pixel_features_rgb_xy[:, :3] = pixel_features_rgb
pixel_features_rgb_xy[:, 3] = normalized_x.flatten()
pixel_features_rgb_xy[:, 4] = normalized_y.flatten()

pixel_features_rgb.shape, pixel_features_rgb_xy.shape

"""# Superpixel-Level Features"""

from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.filters import gabor_kernel
from scipy.ndimage import convolve

def extract_superpixel_features(image, num_segments=20, compactness=10, num_bins=256):
    # SLIC and obtain the segment labels
    segments = slic(image, n_segments=num_segments, compactness=compactness)
    num_superpixels = len(np.unique(segments))

    mean_rgb_values = np.zeros((num_superpixels, 3))
    rgb_histograms = np.zeros((num_superpixels, 3 * num_bins))
    gabor_responses = []

    # Gabor filter bank with different frequencies and orientations
    frequencies = [0.1, 0.2, 0.4, 0.8]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_kernels = [gabor_kernel(frequency, theta=theta) for frequency in frequencies for theta in thetas]

    # grayscale for Gabor filtering
    gray_image = rgb2gray(image)
    for kernel in gabor_kernels:
        gabor_response = convolve(gray_image, kernel.real)
        gabor_responses.append(gabor_response)

    for i, segment_value in enumerate(np.unique(segments)):
        mask = segments == segment_value

        # Mean RGB values
        mean_rgb_values[i, :] = np.mean(image[mask], axis=0)

        # RGB Histograms
        for j in range(3):  # For each RGB channel
            hist, _ = np.histogram(image[mask][:, j], bins=num_bins, range=(0, 256))
            rgb_histograms[i, j * num_bins:(j + 1) * num_bins] = hist

        # Mean Gabor responses
        mean_gabor_response = np.array([np.mean(response[mask]) for response in gabor_responses])
        if i == 0:
            gabor_features = np.zeros((num_superpixels, len(mean_gabor_response)))
        gabor_features[i, :] = mean_gabor_response

    return mean_rgb_values, rgb_histograms, gabor_features, segments

mean_rgb_values, rgb_histograms, gabor_features, segments = extract_superpixel_features(img)

plt.imshow(segments)
plt.title("Superpixels")
plt.axis('off')
plt.show()

from skimage.segmentation import mark_boundaries
from skimage import img_as_float

def resize_image(image, max_dimension=800):
    max_current_dimension = max(image.shape[:2])
    scaling_factor = max_dimension / max_current_dimension
    if scaling_factor < 1.0:
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return image

img = img_as_float(img)

num_segments = 50
# Re-apply SLIC to the original image to obtain superpixel boundaries
segments_slic = slic(img, n_segments=num_segments, compactness=10)

fig, axes = plt.subplots(1, 2, figsize=(24, 12))
ax = axes.ravel()

ax[0].imshow(img)
ax[0].set_title('a) Input Image')
ax[0].axis('off')

ax[1].imshow(mark_boundaries(img, segments_slic))
ax[1].set_title('b) Superpixels')
ax[1].axis('off')

plt.tight_layout()
plt.show()

# K-Means clustering
def k_means_clustering(data, k, max_iterations=100, tolerance=0.0001):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iterations):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        nearest_centroid = np.argmin(distances, axis=0)
        new_centroids = np.array([data[nearest_centroid == i].mean(axis=0) for i in range(k)])
        if np.sum((new_centroids - centroids)**2) < tolerance:
            break

        centroids = new_centroids

    return nearest_centroid, centroids

img = cv2.imread('image1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

normalized_rgb = img / 255.0
height, width, _ = img.shape
normalized_x = np.tile(np.arange(width) / width, (height, 1))
normalized_y = np.tile(np.arange(height) / height, (width, 1)).T

pixel_features_rgb = normalized_rgb.reshape(-1, 3)
pixel_features_rgb_xy = np.zeros((height * width, 5))
pixel_features_rgb_xy[:, :3] = pixel_features_rgb
pixel_features_rgb_xy[:, 3] = normalized_x.flatten()
pixel_features_rgb_xy[:, 4] = normalized_y.flatten()


k = 15

labels, centroids = k_means_clustering(pixel_features_rgb_xy, k)

segmented_image = labels.reshape(height, width)
plt.imshow(segmented_image)
plt.title("K-Means clustering")
plt.axis('off')
plt.show()

from skimage.filters import gabor
def k_means_clustering(data, k, max_iterations=100, tolerance=0.0001):
    if len(data) < k:
        raise ValueError("Number of clusters, k, is greater than the number of data points.")

    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iterations):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        nearest_centroid = np.argmin(distances, axis=1)
        new_centroids = np.array([data[nearest_centroid == i].mean(axis=0) if np.any(nearest_centroid == i) else centroids[i] for i in range(k)])

        if np.allclose(new_centroids, centroids, atol=tolerance):
            break
        centroids = new_centroids

    return nearest_centroid

# Mean RGB Color Values
def calculate_mean_rgb(image, segments):
    mean_rgb = []
    for segment_label in np.unique(segments):
        mask = segments == segment_label
        mean_rgb.append(image[mask].mean(axis=0))
    return np.array(mean_rgb)

# RGB Color Histograms
def calculate_rgb_histograms(image, segments, num_bins=256):
    histograms = []
    for segment_label in np.unique(segments):
        mask = segments == segment_label
        hist = [np.histogram(image[mask][:, i], bins=num_bins, range=(0, 255))[0] for i in range(3)]
        histograms.append(np.concatenate(hist))
    return np.array(histograms)

#  Mean Gabor Filter Responses
def calculate_gabor_features(image, segments, frequencies=[0.1, 0.2, 0.4, 0.8], thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    gray_image = rgb2gray(image)
    gabor_features = []

    # Gabor filters
    for frequency in frequencies:
        for theta in thetas:
            real, imag = gabor(gray_image, frequency=frequency, theta=theta)
            gabor_features.append(real)
            gabor_features.append(imag)

    # mean Gabor response for each superpixel
    features = []
    for segment_label in np.unique(segments):
        mask = segments == segment_label
        segment_features = []
        for response in gabor_features:
            segment_features.append(np.mean(response[mask]))
        features.append(segment_features)

    return np.array(features)

img_path = 'image1.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = img_as_float(img)

# superpixels using SLIC
num_segments = 350
segments_slic = slic(img_float, n_segments=num_segments, compactness=10)


mean_rgb = calculate_mean_rgb(img_float, segments_slic)
rgb_histograms = calculate_rgb_histograms(img_float, segments_slic)
mean_gabor = calculate_gabor_features(img_float, segments_slic)

labels_mean_rgb = k_means_clustering(mean_rgb, k=5)
labels_rgb_histograms = k_means_clustering(rgb_histograms, k=5)
labels_mean_gabor = k_means_clustering(mean_gabor, k=5)

def reshape_labels_to_image(segments, labels):
    reshaped_labels = np.zeros_like(segments)
    unique_segments = np.unique(segments)
    for segment_label, cluster_label in zip(unique_segments, labels):
        reshaped_labels[segments == segment_label] = cluster_label
    return reshaped_labels

labels_mean_rgb_reshaped = reshape_labels_to_image(segments_slic, labels_mean_rgb)
labels_rgb_histograms_reshaped = reshape_labels_to_image(segments_slic, labels_rgb_histograms)
labels_mean_gabor_reshaped = reshape_labels_to_image(segments_slic, labels_mean_gabor)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(mark_boundaries(img_float, labels_mean_rgb_reshaped))
ax[0].set_title('Mean RGB Color Values')
ax[0].axis('off')

ax[1].imshow(mark_boundaries(img_float, labels_rgb_histograms_reshaped))
ax[1].set_title('RGB Color Histograms')
ax[1].axis('off')

ax[2].imshow(mark_boundaries(img_float, labels_mean_gabor_reshaped))
ax[2].set_title('Mean Gabor Filter Responses')
ax[2].axis('off')

plt.show()

def calculate_pixel_rgb(image):
    return image.reshape(-1, 3)

def calculate_pixel_rgb_histograms(image, num_bins=256):
    histograms = np.zeros((image.shape[0] * image.shape[1], 3 * num_bins))
    for i in range(3):  # For each color channel
        hist, _ = np.histogram(image[:, :, i].flatten(), bins=num_bins, range=(0, 1))
        histograms[:, i * num_bins:(i + 1) * num_bins] = hist
    return histograms

# pixel-level RGB histogram features
pixel_rgb_histograms = calculate_pixel_rgb_histograms(img_float)

# K-means clustering
labels_rgb_histograms = k_means_clustering(pixel_rgb_histograms, k=3)


# pixel-level Gabor features
def calculate_pixel_gabor_features(image, frequencies=[0.1, 0.2, 0.4, 0.8]):
    gray_image = rgb2gray(image)
    gabor_features = []

    for frequency in frequencies:
        real, imag = gabor(gray_image, frequency=frequency)
        gabor_features.append(real.flatten())
        gabor_features.append(imag.flatten())

    return np.array(gabor_features).T  # Transpose so that each row is a pixel's feature

img_path = 'image1.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = img_as_float(img)

pixel_rgb = calculate_pixel_rgb(img_float)
pixel_rgb_histograms = calculate_pixel_rgb_histograms(img_float)
pixel_gabor = calculate_pixel_gabor_features(img_float)
labels_rgb = k_means_clustering(pixel_rgb, k=3)
labels_rgb_histograms = k_means_clustering(pixel_rgb_histograms, k=3)
labels_gabor = k_means_clustering(pixel_gabor, k=5)
labels_rgb_reshaped = labels_rgb.reshape(img_float.shape[:2])
labels_rgb_histograms_reshaped = labels_rgb_histograms.reshape(img_float.shape[:2])
labels_gabor_reshaped = labels_gabor.reshape(img_float.shape[:2])

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(labels_rgb_reshaped)
ax[0].set_title('RGB Color Features')
ax[0].axis('off')

ax[1].imshow(labels_rgb_histograms_reshaped)
ax[1].set_title('RGB Color Histograms')
ax[1].axis('off')

ax[2].imshow(labels_gabor_reshaped)
ax[2].set_title('Gabor Filter Responses')
ax[2].axis('off')

plt.show()

from skimage.color import label2rgb

def k_means_clustering(data, k, max_iterations=100, tolerance=0.0001):
    n_samples, _ = data.shape
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[random_indices]
    labels = np.zeros(n_samples, dtype= int)
    distances = np.zeros((n_samples, k))
    for iteration in range(max_iterations):
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.linalg.norm(data - centroid, axis=1)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for i in range(k):
            assigned_data = data[labels == i]
            if assigned_data.size:
                centroids[i] = assigned_data.mean(axis=0)
            else:
                centroids[i] = data[np.random.randint(0, n_samples)]
    return labels

# Mean RGB Color Values
def calculate_mean_rgb(image):
    return image.reshape(-1, 3)

# RGB Color Histograms
def calculate_rgb_histograms(image, num_bins=256):
    histograms = np.zeros((image.shape[0] * image.shape[1], 3 * num_bins))
    for i in range(3):
        hist, _ = np.histogram(image[:, :, i].flatten(), bins=num_bins, range=(0, 1))
        histograms[:, i * num_bins:(i + 1) * num_bins] = hist
    return histograms

# Gabor Filter Responses
def calculate_gabor_features(image, frequencies=[0.1, 0.2, 0.4, 0.8]):
    gray_image = rgb2gray(image)
    gabor_features = []
    for frequency in frequencies:
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            real, imag = gabor(gray_image, frequency=frequency, theta=theta)
            gabor_features.append(real.flatten())
            gabor_features.append(imag.flatten())
    return np.array(gabor_features).T

img_path = 'image1.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = img_as_float(img)

num_segments = 300
segments_slic = slic(img_float, n_segments=num_segments, compactness=10)

mean_rgb = calculate_mean_rgb(img_float)
rgb_histograms = calculate_rgb_histograms(img_float)
gabor_features = calculate_gabor_features(img_float)

k = 7
labels_rgb = k_means_clustering(mean_rgb, k)
labels_histograms = k_means_clustering(rgb_histograms, k)
labels_gabor = k_means_clustering(gabor_features, k)
labels_rgb_reshaped = labels_rgb.reshape(img_float.shape[:2])
labels_histograms_reshaped = labels_histograms.reshape(img_float.shape[:2])
labels_gabor_reshaped = labels_gabor.reshape(img_float.shape[:2])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(label2rgb(labels_rgb_reshaped, image=img_float))
axes[0].set_title('Mean RGB Color Values')
axes[0].axis('off')

axes[1].imshow(label2rgb(labels_histograms_reshaped, image=img_float))
axes[1].set_title('RGB Color Histograms')
axes[1].axis('off')

axes[2].imshow(label2rgb(labels_gabor_reshaped, image=img_float))
axes[2].set_title('Gabor Filter Responses')
axes[2].axis('off')

plt.tight_layout()
plt.show()

img_path = 'image1.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img_as_float(img)

def k_means_clustering(data, k, max_iterations=100, tolerance=0.0001):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iterations):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        nearest_centroid = np.argmin(distances, axis=0)
        new_centroids = np.array([data[nearest_centroid == i].mean(axis=0) for i in range(k)])
        if np.sum((new_centroids - centroids)**2) < tolerance:
            break
        centroids = new_centroids
    return nearest_centroid, centroids

k = 15
labels_rgb, centroids_rgb = k_means_clustering(pixel_features_rgb, k)

# K-Means Clustering on pixel-level RGB+XY features
labels_rgb_xy, centroids_rgb_xy = k_means_clustering(pixel_features_rgb_xy, k)

segmented_image_rgb = labels_rgb.reshape(height, width)
segmented_image_rgb_xy = labels_rgb_xy.reshape(height, width)

fig, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].imshow(segmented_image_rgb)
axes[0].set_title('Segmentation using RGB features')
axes[0].axis('off')

axes[1].imshow(segmented_image_rgb_xy)
axes[1].set_title('Segmentation using RGB+XY features')
axes[1].axis('off')

plt.tight_layout()
plt.show()

def k_means_clustering(data, k, max_iterations=100, tolerance=0.0001):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    prev_centroids = centroids.copy()
    for _ in range(max_iterations):
        distances = np.linalg.norm(data - centroids[:, np.newaxis], axis=2)
        nearest_centroid = np.argmin(distances, axis=0)
        centroids = np.array([data[nearest_centroid == i].mean(axis=0) for i in range(k)])
        shift = np.linalg.norm(centroids - prev_centroids, axis=1).sum()
        if shift < tolerance:
            break
        prev_centroids = centroids.copy()

    return nearest_centroid, centroids

img_path = 'image1.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img_as_float(img)
img_resized = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

num_segments = 70  # Reduce the number of segments for less memory usage
segments_slic = slic(img_resized, n_segments=num_segments, compactness=10.0, start_label=1)
superpixel_features = np.array([img_resized[segments_slic == n].mean(axis=0) if img_resized[segments_slic == n].size > 0 else np.zeros(3) for n in range(1, num_segments + 1)])

k = 5  # Reduce the number of clusters for less memory usage
labels, _ = k_means_clustering(superpixel_features, k)

# Mapping the K-Means labels back to the original image size
segmented_image = np.zeros_like(segments_slic, dtype=int)
for seg_label, kmeans_label in zip(np.unique(segments_slic), labels):
    segmented_image[segments_slic == seg_label] = kmeans_label
overlayed_image = mark_boundaries(img_resized, segmented_image, color=(1, 1, 1), mode='thick')

plt.figure(figsize=(10, 8))
plt.imshow(overlayed_image)
plt.title("d) Image Segments")
plt.axis('off')
plt.show()