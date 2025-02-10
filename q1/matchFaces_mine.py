# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import warnings
warnings.filterwarnings("ignore")

##########################################
# a) scale_down(image, resize_ratio)
##########################################
def scale_down(image, resize_ratio):
    """
    Scale down the image using a Fourier transform (no smoothing).
    The output dimensions are (resize_ratio * original dimensions).
    """
    image_float = image.astype(np.float32)
    orig_h, orig_w = image.shape
    new_h = int(orig_h * resize_ratio)
    new_w = int(orig_w * resize_ratio)

    F = fftshift(fft2(image_float))
    center_y, center_x = F.shape[0] // 2, F.shape[1] // 2
    half_new_h = new_h // 2
    half_new_w = new_w // 2

    if new_h % 2 == 0:
        y_start = center_y - half_new_h
        y_end = center_y + half_new_h
    else:
        y_start = center_y - half_new_h
        y_end = center_y + half_new_h + 1

    if new_w % 2 == 0:
        x_start = center_x - half_new_w
        x_end = center_x + half_new_w
    else:
        x_start = center_x - half_new_w
        x_end = center_x + half_new_w + 1

    F_cropped = F[y_start:y_end, x_start:x_end]
    result = ifft2(ifftshift(F_cropped))
    result = np.abs(result)
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

##########################################
# b) scale_up(image, resize_ratio)
##########################################
def scale_up(image, resize_ratio):
    """
    Scale up the image using a Fourier transform.
    Zero-pad the Fourier spectrum so that the output dimensions are
    (resize_ratio * original dimensions).
    """
    image_float = image.astype(np.float32)
    orig_h, orig_w = image.shape
    new_h = int(orig_h * resize_ratio)
    new_w = int(orig_w * resize_ratio)

    F = fftshift(fft2(image_float))
    newF = np.zeros((new_h, new_w), dtype=complex)

    start_y = (new_h - orig_h) // 2
    start_x = (new_w - orig_w) // 2
    newF[start_y:start_y+orig_h, start_x:start_x+orig_w] = F

    result = ifft2(ifftshift(newF))
    result = np.abs(result)
    scale_factor = (new_h * new_w) / (orig_h * orig_w)
    result = result * scale_factor
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

def scale_image(image, resize_ratio):
    """
    Uniformly scale an image using Fourier methods.
    """
    if resize_ratio < 1:
        return scale_down(image, resize_ratio)
    elif resize_ratio > 1:
        return scale_up(image, resize_ratio)
    else:
        return image.copy()

##########################################
# c) ncc_2d(image, pattern)
##########################################
def ncc_2d(image, pattern):
    """
    Compute normalized cross-correlation (NCC) between a grayscale image and a pattern.
    For memory efficiency we use OpenCV's matchTemplate with TM_CCORR_NORMED.
    (This avoids creating a huge sliding_window_view array.)
    """
    # cv2.matchTemplate returns an array of size (image_h - pattern_h + 1, image_w - pattern_w + 1)
    result = cv2.matchTemplate(image, pattern, cv2.TM_CCORR_NORMED)
    return result.astype(np.float32)

##########################################
# d) display(image, pattern)
##########################################
def display(image, pattern):
    """
    Display the image, the pattern, and the squared NCC heatmap.
    """
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(image, cmap="gray", aspect="equal")

    plt.subplot(1,3,2)
    plt.title("Pattern")
    plt.imshow(pattern, cmap="gray", aspect="equal")

    ncc = ncc_2d(image, pattern)
    plt.subplot(1,3,3)
    plt.title("NCC Heatmap (squared)")
    plt.imshow(ncc**2, cmap="coolwarm", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(label="NCC Values")
    plt.show()

##########################################
# e) draw_matches(image, matches, pattern_size)
##########################################
def draw_matches(image, matches, pattern_size):
    """
    Draw red rectangles on the original color image at the detected match centers.
    """
    out_img = image.copy()
    for (y, x) in matches:
        top_left = (int(x - pattern_size[1] // 2), int(y - pattern_size[0] // 2))
        bottom_right = (int(x + pattern_size[1] // 2), int(y + pattern_size[0] // 2))
        cv2.rectangle(out_img, top_left, bottom_right, (0, 0, 255), 2)
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Matches")
    plt.axis("off")
    plt.show()
    cv2.imwrite("result.jpg", out_img)

##########################################
# Main Script
##########################################
# We process two images: "students.jpg" and "thecrew.jpg"
# We have one template "template.jpg". The instructions indicate:
#   - For students.jpg: expected face ~40px high, 28px wide.
#   - For thecrew.jpg: expected face ~7px high, 5px wide.
# We compute a scale factor so that the faces in the image become approximately
# the same size as the template.
# Scale factor = average( template_dimension / expected_face_dimension )
# (Then we scale the image by that factor, compute NCC, and map match coordinates back.)

threshold = 0.35  # Use a 0.35 threshold

# -----------------------------
# Process Students Image
# -----------------------------
CURR_IMAGE = "students"
students_color = cv2.imread("students.jpg")
students_gray = cv2.cvtColor(students_color, cv2.COLOR_BGR2GRAY)
template_img = cv2.imread("template.jpg")
template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

# Compute scale factor:
#   scale_factor = average( template_height/40, template_width/28 )
scale_factor_students = ((template_gray.shape[0] / 40) + (template_gray.shape[1] / 28)) / 2.0
print("Students scale factor:", scale_factor_students)

students_scaled = scale_image(students_gray, scale_factor_students)
# We use the template as-is.
students_pattern = template_gray.copy()

display(students_scaled, students_pattern)

ncc_students = ncc_2d(students_scaled, students_pattern)
matches_students = np.argwhere(ncc_students > threshold)
# Adjust coordinates to center of pattern.
matches_students[:, 0] += students_pattern.shape[0] // 2
matches_students[:, 1] += students_pattern.shape[1] // 2
# Map coordinates back to original image space.
matches_students = np.floor(matches_students / scale_factor_students).astype(int)

draw_matches(students_color, matches_students, students_pattern.shape)

# -----------------------------
# Process Crew Image
# -----------------------------
CURR_IMAGE = "thecrew"
crew_color = cv2.imread("thecrew.jpg")
crew_gray = cv2.cvtColor(crew_color, cv2.COLOR_BGR2GRAY)
template_img = cv2.imread("template.jpg")
template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

# For thecrew.jpg: expected face ~7px high, 5px wide.
scale_factor_crew = ((template_gray.shape[0] / 7) + (template_gray.shape[1] / 5)) / 2.0
print("Crew scale factor:", scale_factor_crew)

crew_scaled = scale_image(crew_gray, scale_factor_crew)
crew_pattern = template_gray.copy()  # Template remains unscaled

display(crew_scaled, crew_pattern)

ncc_crew = ncc_2d(crew_scaled, crew_pattern)
matches_crew = np.argwhere(ncc_crew > threshold)
matches_crew[:, 0] += crew_pattern.shape[0] // 2
matches_crew[:, 1] += crew_pattern.shape[1] // 2
# Map match coordinates back to the original image.
matches_crew = np.floor(matches_crew / scale_factor_crew).astype(int)

draw_matches(crew_color, matches_crew, crew_pattern.shape)
