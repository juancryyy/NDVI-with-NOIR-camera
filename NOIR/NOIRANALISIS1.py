import cv2
import numpy as np
import matplotlib.pyplot as plt

def calibrate_image(image):
    """Normalizes the image to improve contrast."""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def split_image(image):
    """Assumes the image has 3 channels: B, G, R."""
    b, g, r = cv2.split(image)
    return r, g, b  # Return Red, Green, Blue channels

def calc_ndvi(nir_image, red_image):
    """Calculates NDVI using NIR and Red channels."""
    # Avoid division by zero by adding a small constant
    bottom = (nir_image.astype(float) + red_image.astype(float))
    bottom[bottom == 0] = 0.01
    ndvi = (nir_image.astype(float) - red_image.astype(float)) / bottom
    return ndvi

def apply_color_map(ndvi):
    """Applies a color map to NDVI values for visualization."""
    ndvi_min = np.min(ndvi)
    ndvi_max = np.max(ndvi)
    ndvi_normalized = (ndvi - ndvi_min) / (ndvi_max - ndvi_min)  # Normalize to [0, 1]
    ndvi_normalized = (ndvi_normalized * 255).astype(np.uint8)
    color_mapped_image = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
    return color_mapped_image

def convert_to_gray(ndvi):
    """Converts NDVI values to grayscale."""
    ndvi_min = np.min(ndvi)
    ndvi_max = np.max(ndvi)
    ndvi_normalized = (ndvi - ndvi_min) / (ndvi_max - ndvi_min)
    ndvi_gray = (ndvi_normalized * 255).astype(np.uint8)
    return ndvi_gray

# Load images
natural_image = cv2.imread('NATURAL.WEBP')  # RGB image
nir_image = cv2.imread('RED.WEBP', cv2.IMREAD_UNCHANGED)  # Assuming it's a single channel NIR image

# Calibrate images
calibrated_natural = calibrate_image(natural_image)

# Split RGB image into channels
r_channel, g_channel, b_channel = split_image(calibrated_natural)

# Ensure the NIR image is in the right format
if len(nir_image.shape) == 3:
    nir_image = cv2.cvtColor(nir_image, cv2.COLOR_BGR2GRAY)  # Convert to single channel if needed

# Calculate NDVI
ndvi = calc_ndvi(nir_image, r_channel)

# Convert NDVI to grayscale
ndvi_gray = convert_to_gray(ndvi)

# Apply color map for visualization
color_mapped_image = apply_color_map(ndvi)

# Display the images using matplotlib
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("NIR Image")
plt.imshow(nir_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Visible Red Image")
plt.imshow(r_channel, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("NDVI Color Mapped")
plt.imshow(color_mapped_image)
plt.axis('off')

# Add color scale
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=plt.gca(), orientation='vertical')
cbar.set_label('NDVI Scale')

plt.subplot(2, 3, 4)
plt.title("NDVI Grayscale")
plt.imshow(ndvi_gray, cmap='gray')
plt.axis('off')

# Export results
cv2.imwrite('ndvi_color_mapped_image.png', color_mapped_image)
cv2.imwrite('ndvi_gray_image.png', ndvi_gray)

# Export separated channels
cv2.imwrite('red_channel_image.png', r_channel)
cv2.imwrite('green_channel_image.png', g_channel)
cv2.imwrite('blue_channel_image.png', b_channel)
cv2.imwrite('nir_image.png', nir_image)

plt.tight_layout()
plt.show()
