# Idan Morad, 31645101
# Nadav Melman, 206171548

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import warnings

warnings.filterwarnings("ignore")

# global variable used for scaling
CURR_IMAGE = "students"


def scale_down(image, resize_ratio):
    # downscale the image using the fourier domain
    if not (0 < resize_ratio < 1):
        raise ValueError("resize_ratio must be between 0 and 1")
    h, w = image.shape[:2]
    new_h = int(round(h * resize_ratio))
    new_w = int(round(w * resize_ratio))
    f_transform = fft2(image, axes=(0, 1))
    f_shifted = fftshift(f_transform, axes=(0, 1))
    center_h, center_w = h // 2, w // 2
    half_new_h = new_h // 2
    half_new_w = new_w // 2
    if new_h % 2 == 0:
        start_h = center_h - half_new_h
        end_h = center_h + half_new_h
    else:
        start_h = center_h - half_new_h
        end_h = center_h + half_new_h + 1
    if new_w % 2 == 0:
        start_w = center_w - half_new_w
        end_w = center_w + half_new_w
    else:
        start_w = center_w - half_new_w
        end_w = center_w + half_new_w + 1
    if image.ndim == 2:
        f_cropped = f_shifted[start_h:end_h, start_w:end_w]
    else:
        f_cropped = f_shifted[start_h:end_h, start_w:end_w, :]
    f_uncentered = ifftshift(f_cropped, axes=(0, 1))
    img_small = ifft2(f_uncentered, axes=(0, 1))
    img_small = np.real(img_small)
    scaling_factor = (new_h * new_w) / (h * w)
    img_small *= scaling_factor
    return img_small


def scale_up(image, resize_ratio):
    # upscale the image using the fourier transform
    if resize_ratio < 1:
        raise ValueError("resize_ratio must be at least 1")
    h, w = image.shape[:2]
    #calc new dimensions
    new_h = int(round(h * resize_ratio))
    new_w = int(round(w * resize_ratio))
    f_transform = fft2(image, axes=(0, 1))
    f_shifted = fftshift(f_transform, axes=(0, 1))

    #bring in our faorite global dude
    global CURR_IMAGE
    #thecrew gets aliased so we need to fix by smoothing
    if CURR_IMAGE == "thecrew":
        # for "thecrew", apply a frequency mask to reduce aliasing
        y = np.arange(h) - h / 2
        x = np.arange(w) - w / 2
        X, Y = np.meshgrid(x, y)
        d = np.sqrt(X ** 2 + Y ** 2)
        cutoff = 0.420 * min(h, w)
        mask = (d < cutoff).astype(f_shifted.dtype)
        f_shifted = f_shifted * mask
    #since thecrew is colored and we want the result to be grayscale with colored rectangles
    #we need to make sure it has color channels
    if image.ndim == 2:
        f_padded = np.zeros((new_h, new_w), dtype=f_shifted.dtype)
    else:
        channels = image.shape[2]
        f_padded = np.zeros((new_h, new_w, channels), dtype=f_shifted.dtype)
    pad_top = (new_h - h) // 2
    pad_left = (new_w - w) // 2
    if image.ndim == 2:
        f_padded[pad_top:pad_top + h, pad_left:pad_left + w] = f_shifted
    else:
        f_padded[pad_top:pad_top + h, pad_left:pad_left + w, :] = f_shifted
    f_uncentered = ifftshift(f_padded, axes=(0, 1))
    img_large = ifft2(f_uncentered, axes=(0, 1))
    img_large = np.real(img_large)
    scaling_factor = (new_h * new_w) / (h * w)
    img_large *= scaling_factor
    return img_large


def ncc_2d(image, pattern):
    # compute normalized cross-correlation between image and pattern
    image = np.asarray(image, dtype=np.float32)
    pattern = np.asarray(pattern, dtype=np.float32)
    ph, pw = pattern.shape
    pad_top = (ph - 1) // 2
    pad_bottom = ph - 1 - pad_top
    pad_left = (pw - 1) // 2
    pad_right = pw - 1 - pad_left
    padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    windows = np.lib.stride_tricks.sliding_window_view(padded, pattern.shape)
    window_mean = np.mean(windows, axis=(-2, -1))
    windows_zero = windows - window_mean[..., None, None]
    pattern_mean = np.mean(pattern)
    pattern_zero = pattern - pattern_mean
    numer = np.sum(windows_zero * pattern_zero, axis=(-2, -1))
    win_norm = np.sqrt(np.sum(windows_zero ** 2, axis=(-2, -1)))
    pat_norm = np.sqrt(np.sum(pattern_zero ** 2))
    denom = win_norm * pat_norm
    ncc = np.where(denom != 0, numer / denom, 0)
    return ncc


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
    plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('NCC Values')

    plt.show()


def draw_matches(image, matches, pattern_size):
    # convert the image to uint8 and then to BGR for drawing
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    rgb_img = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    for point in matches:
        y, x = point  # (row, column)
        top_left = (int(x - pattern_size[1] / 2), int(y - pattern_size[0] / 2))
        bottom_right = (int(x + pattern_size[1] / 2), int(y + pattern_size[0] / 2))
        # draw red rectangle (bgr: (0, 0, 255))
        cv2.rectangle(rgb_img, top_left, bottom_right, (0, 0, 255), 1)

    plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    cv2.imwrite(f"{CURR_IMAGE}_result.jpg", rgb_img)


def apply_gamma_correction(image, gamma):
    # apply gamma correction: output = 255 * (image/255)^(1/gamma)
    norm_img = image / 255.0
    gamma_corr = np.power(norm_img, 1.0 / gamma) * 255.0
    return np.clip(gamma_corr, 0, 255).astype(np.float32)


def process_image(img, brightness, contrast, gamma):
    #  process the image: adjust brightness, contrast, then apply gamma correction
    proc = np.clip(img * brightness, 0, 255).astype(np.float32)
    proc = np.clip(proc * contrast, 0, 255).astype(np.float32)
    proc = apply_gamma_correction(proc, gamma)
    return proc


def process_template(template):
    #  process the template (no brightness/contrast/gamma changes)
    return template.astype(np.float32)


CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############# DEMO #############
display(image, pattern)

############# students.jpg #############

image_scaled = scale_up(image, 1.32)
pattern_scaled = pattern

display(image_scaled, pattern_scaled)

processed_img = process_image(image_scaled, 1.9, 0.7, 0.21)
processed_img = cv2.GaussianBlur(processed_img, (7, 7), 2)
processed_pattern = process_template(pattern_scaled)
processed_pattern = cv2.GaussianBlur(processed_pattern, (7, 7), 2)
ncc = ncc_2d(processed_img, processed_pattern)
real_matches = np.argwhere(ncc > 0.408)

draw_matches(image_scaled, real_matches, pattern_scaled.shape)

############# thecrew.jpg #############

image_scaled = scale_up(cv2.cvtColor(cv2.imread("thecrew.jpg"), cv2.COLOR_BGR2GRAY), 3.33)
pattern_scaled = pattern

display(image_scaled, pattern_scaled)

processed_img = process_image(image_scaled, 1.9, 0.7, 0.21)
processed_img = cv2.GaussianBlur(processed_img, (7, 7), 2)
processed_pattern = process_template(pattern_scaled)
processed_pattern = cv2.GaussianBlur(processed_pattern, (7, 7), 2)
ncc = ncc_2d(processed_img, processed_pattern)
real_matches = np.argwhere(ncc > 0.408)

draw_matches(image_scaled, real_matches, pattern_scaled.shape)
