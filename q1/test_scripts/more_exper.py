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
THRESHOLD = 0.42     # NCC threshold (set to 0.35)
BLUR_KERNEL = (7, 7)     # Kernel size for Gaussian blur
BLUR_SIGMA = 2.0         # Sigma value for Gaussian blur (adjust as desired)

# Fixed rectangle size: 37x37 (height, width)
RECT_SIZE = (37, 37)

####################################
# Fourier-domain Scaling Functions #
####################################

def scale_down(image, resize_ratio):
    """Downscale an image by a factor (0 < resize_ratio < 1) using the Fourier domain."""
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
    """Upscale an image by a factor (>= 1) using the Fourier domain."""
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
        cutoff = 0.420 * min(H, W)
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

#####################################
# Normalized Cross-Correlation (NCC)#
#####################################

def ncc_2d(image, pattern):
    """
    Compute the normalized cross-correlation (NCC) between a given pattern (kernel)
    and every window in the image of the same size as the pattern.
    """
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

#########################
# Display Function      #
#########################

def display(image, pattern):
    # Original 2x3 subplot layout for image, pattern, and squared NCC heatmap.
    plt.subplot(2, 3, 1)
    plt.title('Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Pattern')
    plt.imshow(pattern, cmap='gray', aspect='equal')

    ncc = ncc_2d(image, pattern)

    plt.subplot(2, 3, 5)
    plt.title('Squared NCC Heatmap')
    plt.imshow(ncc**2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Squared NCC Values')
    plt.show()

#########################
# Draw Matches Function #
#########################

def draw_matches(image, matches, rect_size):
    """
    Convert the image to color, then draw red rectangles (BGR: (0,0,255))
    at each match location. The rectangle is of size rect_size (e.g. 37x37) and is centered at the match.
    """
    # Convert the grayscale image to color for drawing red rectangles.
    image_color = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for pt in matches:
        y, x = pt  # y is row, x is column
        top_left = (int(x - rect_size[1] // 2), int(y - rect_size[0] // 2))
        bottom_right = (int(x + rect_size[1] // 2), int(y + rect_size[0] // 2))
        cv2.rectangle(image_color, top_left, bottom_right, (0, 0, 255), 1)  # Red rectangle in BGR
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.title("Matches drawn on image")
    plt.axis("off")
    plt.show()
    cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image_color)

#######################
# Main Workflow       #
#######################

# --- Process "students.jpg" ---
CURR_IMAGE = "students"
img_students = cv2.imread(f"{CURR_IMAGE}.jpg", cv2.IMREAD_GRAYSCALE)
if img_students is None:
    raise IOError("Could not load 'students.jpg'.")

# Increase contrast using convertScaleAbs (alpha=1.2)
img_students = cv2.convertScaleAbs(img_students, alpha=1.2, beta=0)

# Enlarge using scale factor 1.32 for students
scale_factor_students = 1.32
img_students_scaled = scale_up(img_students, scale_factor_students)

# Apply Gaussian blur on the enlarged image BEFORE NCC calculation.
img_students_scaled = cv2.GaussianBlur(img_students_scaled.astype(np.float32), BLUR_KERNEL, BLUR_SIGMA*0.9)

# Read the template in grayscale and apply Gaussian blur on the pattern.
template = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)
if template is None:
    raise IOError("Could not load 'template.jpg'.")
template = cv2.GaussianBlur(template.astype(np.float32), BLUR_KERNEL, BLUR_SIGMA)

# --------------------------------
# Apply Laplacian to both images, fix unsupported src/dst error
# --------------------------------

students_laplacian = cv2.Laplacian(img_students_scaled.astype(np.float32), cv2.CV_32F)
students_laplacian = cv2.convertScaleAbs(students_laplacian)

pattern_laplacian = cv2.Laplacian(template.astype(np.float32), cv2.CV_32F)
pattern_laplacian = cv2.convertScaleAbs(pattern_laplacian)

# Display the Laplacian images and squared NCC heatmap
display(students_laplacian, pattern_laplacian)

# Compute NCC on the Laplacian images
ncc_students = ncc_2d(students_laplacian, pattern_laplacian)
# Find match coordinates where the raw NCC > THRESHOLD.
matches_students = np.argwhere(ncc_students > THRESHOLD)

# Draw matches on the scaled image (not Laplacian) for clarity.
draw_matches(img_students_scaled, matches_students, RECT_SIZE)

# --- Process "thecrew.jpg" ---
CURR_IMAGE = "thecrew"
img_thecrew_color = cv2.imread(f"{CURR_IMAGE}.jpg")
if img_thecrew_color is None:
    raise IOError("Could not load 'thecrew.jpg'.")

# Convert to grayscale.
img_thecrew_gray = cv2.cvtColor(img_thecrew_color, cv2.COLOR_BGR2GRAY)

# Increase contrast using convertScaleAbs (alpha=1.2)
img_thecrew_gray = cv2.convertScaleAbs(img_thecrew_gray, alpha=1.2, beta=0)

# Enlarge using scale factor 3.33 for thecrew.
scale_factor_thecrew = 3.33
img_thecrew_scaled = scale_up(img_thecrew_gray, scale_factor_thecrew)

# Apply Gaussian blur on the enlarged image BEFORE NCC calculation.
img_thecrew_scaled = cv2.GaussianBlur(img_thecrew_scaled.astype(np.float32), BLUR_KERNEL, BLUR_SIGMA)

# Reuse the same blurred template => but also recalc Laplacian for clarity:
pattern_laplacian_thecrew = cv2.Laplacian(template.astype(np.float32), cv2.CV_32F)
pattern_laplacian_thecrew = cv2.convertScaleAbs(pattern_laplacian_thecrew)

# Laplacian for the thecrew image
thecrew_laplacian = cv2.Laplacian(img_thecrew_scaled.astype(np.float32), cv2.CV_32F)
thecrew_laplacian = cv2.convertScaleAbs(thecrew_laplacian)

display(thecrew_laplacian, pattern_laplacian_thecrew)
ncc_thecrew = ncc_2d(thecrew_laplacian, pattern_laplacian_thecrew)
matches_thecrew = np.argwhere(ncc_thecrew > THRESHOLD)

draw_matches(img_thecrew_scaled, matches_thecrew, RECT_SIZE)
