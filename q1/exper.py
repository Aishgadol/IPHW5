# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import warnings
warnings.filterwarnings("ignore")

# --------------------------
# Global Parameters
# --------------------------
THRESHOLD = 0.41         # NCC threshold (adjustable)
BLUR_KERNEL = (5, 5)     # Kernel size for Gaussian blur
BLUR_SIGMA = 1.8           # Sigma for Gaussian blur (0 lets OpenCV auto-calc sigma)
RECT_SIZE = (37, 37)     # Rectangle size (height, width)

# --------------------------
# Scaling Functions
# --------------------------
def scale_down(image, resize_ratio):
    # Downscale an image by a factor `resize_ratio` (0 < resize_ratio < 1) using the Fourier domain.
    if not (0 < resize_ratio < 1):
        raise ValueError("For scale_down, resize_ratio must be between 0 and 1.")
    H, W = image.shape[:2]
    new_H = int(round(H * resize_ratio))
    new_W = int(round(W * resize_ratio))
    F = fft2(image, axes=(0, 1))
    F_shifted = fftshift(F, axes=(0, 1))
    center_H, center_W = H//2, W//2
    half_new_H = new_H//2
    half_new_W = new_W//2
    if new_H % 2 == 0:
        start_H = center_H - half_new_H
        end_H = center_H + half_new_H
    else:
        start_H = center_H - half_new_H
        end_H = center_H + half_new_H + 1
    if new_W % 2 == 0:
        start_W = center_W - half_new_W
        end_W = center_W + half_new_W
    else:
        start_W = center_W - half_new_W
        end_W = center_W + half_new_W + 1
    if image.ndim == 2:
        F_cropped = F_shifted[start_H:end_H, start_W:end_W]
    else:
        F_cropped = F_shifted[start_H:end_H, start_W:end_W, :]
    F_cropped = ifftshift(F_cropped, axes=(0, 1))
    img_small = ifft2(F_cropped, axes=(0, 1))
    img_small = np.real(img_small)
    scaling_factor = (new_H * new_W) / (H * W)
    img_small *= scaling_factor
    return img_small

def scale_up(image, resize_ratio):
    # Upscale an image by a factor `resize_ratio` (>= 1) using the Fourier domain.
    if resize_ratio < 1:
        raise ValueError("resize_ratio must be at least 1.")
    H, W = image.shape[:2]
    new_H = int(round(H * resize_ratio))
    new_W = int(round(W * resize_ratio))
    F = fft2(image, axes=(0, 1))
    F_shifted = fftshift(F, axes=(0, 1))

    # For "thecrew", silence high frequencies to mitigate aliasing.
    global CURR_IMAGE
    if CURR_IMAGE == "thecrew":
        y = np.arange(H) - H/2
        x = np.arange(W) - W/2
        X, Y = np.meshgrid(x, y)
        D = np.sqrt(X**2 + Y**2)
        cutoff = 0.35 * min(H, W)  # Adjust this cutoff value if needed.
        mask = (D < cutoff).astype(F_shifted.dtype)
        F_shifted = F_shifted * mask

    if image.ndim == 2:
        F_padded = np.zeros((new_H, new_W), dtype=F_shifted.dtype)
    else:
        channels = image.shape[2]
        F_padded = np.zeros((new_H, new_W, channels), dtype=F_shifted.dtype)
    pad_top = (new_H - H) // 2
    pad_left = (new_W - W) // 2
    if image.ndim == 2:
        F_padded[pad_top:pad_top+H, pad_left:pad_left+W] = F_shifted
    else:
        F_padded[pad_top:pad_top+H, pad_left:pad_left+W, :] = F_shifted
    F_padded = ifftshift(F_padded, axes=(0, 1))
    img_large = ifft2(F_padded, axes=(0, 1))
    img_large = np.real(img_large)
    scaling_factor = (new_H * new_W) / (H * W)
    img_large *= scaling_factor
    return img_large

# --------------------------
# Normalized Cross-Correlation Function
# --------------------------
def ncc_2d(image, pattern):
    # Compute the normalized cross-correlation (NCC) between a given pattern and every window in the image of the same size.
    image = np.asarray(image, dtype=np.float32)
    pattern = np.asarray(pattern, dtype=np.float32)
    ph, pw = pattern.shape
    pad_top = (ph - 1) // 2
    pad_bottom = ph - 1 - pad_top
    pad_left = (pw - 1) // 2
    pad_right = pw - 1 - pad_left
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    windows = np.lib.stride_tricks.sliding_window_view(padded_image, pattern.shape)
    window_mean = np.mean(windows, axis=(-2, -1))
    windows_zero = windows - window_mean[..., None, None]
    pattern_mean = np.mean(pattern)
    pattern_zero = pattern - pattern_mean
    numerator = np.sum(windows_zero * pattern_zero, axis=(-2, -1))
    window_norm = np.sqrt(np.sum(windows_zero**2, axis=(-2, -1)))
    pattern_norm = np.sqrt(np.sum(pattern_zero**2))
    denominator = window_norm * pattern_norm
    ncc = np.where(denominator != 0, numerator / denominator, 0)
    return ncc

# --------------------------
# Display Function
# --------------------------
def display(image, pattern):
    plt.subplot(2, 3, 1)
    plt.title('Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(2, 3, 3)
    plt.title('Pattern')
    plt.imshow(pattern, cmap='gray', aspect='equal')
    ncc = ncc_2d(image, pattern)
    plt.subplot(2, 3, 5)
    plt.title('Normalized Cross-Correlation Heatmap')
    # Display squared NCC values.
    plt.imshow(ncc**2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('NCC Values')
    plt.show()

# --------------------------
# Draw Matches Function
# --------------------------
def draw_matches(image, matches, pattern_size):
    # Convert the grayscale image to color so that red rectangles can be drawn.
    image_color = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for pt in matches:
        y, x = pt  # y is row, x is column.
        top_left = (int(x - pattern_size[1] // 2), int(y - pattern_size[0] // 2))
        bottom_right = (int(x + pattern_size[1] // 2), int(y + pattern_size[0] // 2))
        cv2.rectangle(image_color, top_left, bottom_right, (0, 0, 255), 1)  # Red rectangle.
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.title("Matches drawn on image")
    plt.axis("off")
    plt.show()
    cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image_color)

# --------------------------
# Main Code
# --------------------------

CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############# DEMO #############
display(image, pattern)

############# Students #############
# Increase contrast (alpha=1.2) without brightness change.
image_scaled = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
# Enlarge the students image by a factor of 1.32.
image_scaled = scale_up(image_scaled, 1.32)
# Apply Gaussian blur on the enlarged image BEFORE NCC calculation.
# <-- Adjust BLUR_KERNEL and BLUR_SIGMA here if needed.
image_scaled = cv2.GaussianBlur(image_scaled.astype(np.float32), BLUR_KERNEL, BLUR_SIGMA)
# Apply Gaussian blur on the template (pattern) before NCC calculation.
pattern_scaled = cv2.GaussianBlur(pattern, BLUR_KERNEL, BLUR_SIGMA)
display(image_scaled, pattern_scaled)
ncc = ncc_2d(image_scaled, pattern_scaled)
real_matches = np.argwhere(ncc > THRESHOLD)
# NOTE: The original skeleton suggested adding an offset here,
# but if the rectangles appear shifted right and down, omit this adjustment.
# real_matches[:,0] += pattern_scaled.shape[0] // 2
# real_matches[:,1] += pattern_scaled.shape[1] // 2
draw_matches(image_scaled, real_matches, pattern_scaled.shape)

############# Crew #############
CURR_IMAGE = "thecrew"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
if image is None:
    raise IOError("Could not load 'thecrew.jpg'. Check the file path.")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
# Enlarge the thecrew image by a factor of 3.33.
image_scaled = scale_up(image, 3.33)
# Apply Gaussian blur on the enlarged image BEFORE NCC calculation.
image_scaled = cv2.GaussianBlur(image_scaled.astype(np.float32), BLUR_KERNEL, BLUR_SIGMA)
# Use the same blurred template.
pattern_scaled = cv2.GaussianBlur(pattern, BLUR_KERNEL, BLUR_SIGMA)
display(image_scaled, pattern_scaled)
ncc = ncc_2d(image_scaled, pattern_scaled)
real_matches = np.argwhere(ncc > THRESHOLD)
# Omit additional offset adjustment if it misaligns the rectangles.
# real_matches[:,0] += pattern_scaled.shape[0] // 2
# real_matches[:,1] += pattern_scaled.shape[1] // 2
draw_matches(image_scaled, real_matches, pattern_scaled.shape)
