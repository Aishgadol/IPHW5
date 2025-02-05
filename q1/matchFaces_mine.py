# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def scale_image(image, height_ratio, width_ratio):
    """Scales an image using Fourier Transform based on separate height and width ratios."""
    h, w = image.shape
    new_h, new_w = int(h * height_ratio), int(w * width_ratio)

    print(f"   -> Scaling image to {new_h}x{new_w} (height ratio: {height_ratio:.2f}, width ratio: {width_ratio:.2f})")

    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return scaled_image


def ncc_2d(image, pattern):
    """Computes the Normalized Cross-Correlation (NCC) between an image and a pattern."""
    image_patches = np.lib.stride_tricks.sliding_window_view(image, pattern.shape)
    mean_image = np.mean(image_patches, axis=(-2, -1), keepdims=True)
    mean_pattern = np.mean(pattern)

    numerator = np.sum((image_patches - mean_image) * (pattern - mean_pattern), axis=(-2, -1))
    denominator = np.sqrt(np.sum((image_patches - mean_image) ** 2, axis=(-2, -1)) *
                          np.sum((pattern - mean_pattern) ** 2))

    with np.errstate(divide='ignore', invalid='ignore'):
        ncc = np.where(denominator == 0, 0, numerator / denominator)

    return ncc


def find_matches(image, pattern, threshold=0.6, num_best_matches=10):
    """Finds matches using NCC and returns coordinate list of the brightest spots."""
    ncc = ncc_2d(image, pattern)

    # Find the brightest spots in NCC (top `num_best_matches`)
    flat_indices = np.argsort(ncc.ravel())[::-1]  # Sort in descending order
    best_indices = flat_indices[:num_best_matches]  # Take top matches

    # Convert flat indices to 2D indices
    match_locations = np.array(np.unravel_index(best_indices, ncc.shape)).T

    print(f"   -> Found {len(match_locations)} matches")
    return match_locations


def draw_matches(image, matches, pattern_size):
    """Draws red rectangles at detected matches."""
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color image

    for y, x in matches:
        top_left = (int(x - pattern_size[1] // 2), int(y - pattern_size[0] // 2))
        bottom_right = (int(x + pattern_size[1] // 2), int(y + pattern_size[0] // 2))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)  # Red rectangle

    print(f"-> Drawing {len(matches)} matches on image")

    plt.imshow(image)
    plt.show()

    cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image)
    print(f"-> Saved results as {CURR_IMAGE}_result.jpg")


# Load the pattern
pattern = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
pattern_h, pattern_w = pattern.shape

# STUDENTS IMAGE
CURR_IMAGE = "students"
image = cv2.imread(f'{CURR_IMAGE}.jpg', cv2.IMREAD_GRAYSCALE)

# Range of face sizes to test
height_range_students = range(30, 51)  # 30 to 50 pixels
width_range_students = range(25, 33)  # 25 to 32 pixels

all_matches_students = []

print("\nProcessing students.jpg...")

for face_h in height_range_students:
    for face_w in width_range_students:
        print(f"Testing face size {face_h}x{face_w}")

        height_ratio = pattern_h / face_h
        width_ratio = pattern_w / face_w
        scaled_image = scale_image(image, height_ratio, width_ratio)

        real_matches = find_matches(scaled_image, pattern)

        # Scale back matches to original size
        real_matches[:, 0] = (real_matches[:, 0] / height_ratio).astype(int)
        real_matches[:, 1] = (real_matches[:, 1] / width_ratio).astype(int)

        all_matches_students.extend(real_matches)

# Remove duplicates
all_matches_students = np.unique(all_matches_students, axis=0)

draw_matches(image, all_matches_students, pattern.shape)

# CREW IMAGE
CURR_IMAGE = "thecrew"
image = cv2.imread(f'{CURR_IMAGE}.jpg', cv2.IMREAD_GRAYSCALE)

# Range of face sizes to test
height_range_crew = range(5, 10)  # 5 to 9 pixels
width_range_crew = range(4, 8)  # 4 to 7 pixels

all_matches_crew = []

print("\nProcessing thecrew.jpg...")

for face_h in height_range_crew:
    for face_w in width_range_crew:
        print(f"Testing face size {face_h}x{face_w}")

        height_ratio = pattern_h / face_h
        width_ratio = pattern_w / face_w
        scaled_image = scale_image(image, height_ratio, width_ratio)

        real_matches = find_matches(scaled_image, pattern)

        # Scale back matches to original size
        real_matches[:, 0] = (real_matches[:, 0] / height_ratio).astype(int)
        real_matches[:, 1] = (real_matches[:, 1] / width_ratio).astype(int)

        all_matches_crew.extend(real_matches)

# Remove duplicates
all_matches_crew = np.unique(all_matches_crew, axis=0)

draw_matches(image, all_matches_crew, pattern.shape)
