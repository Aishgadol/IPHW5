import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict

# parametric equation for heart's x coordinate
def heart_x_coordinate(t):
    return 14.5 * np.sin(t) ** 3  # compute x offset

# parametric equation for heart's y coordinate
def heart_y_coordinate(t):
    return 0.5 * np.cos(4 * t) + 2 * np.cos(3 * t) + 4 * np.cos(2 * t) - 13 * np.cos(t)  # compute y offset

# generate a heart contour (centered at (0,0)) using the parametric equations
def generate_heart_contour(scale=1):
    t_vals = np.linspace(0, 2 * np.pi, 300)
    x_vals = (scale * heart_x_coordinate(t_vals)).astype(np.int32)
    y_vals = (scale * heart_y_coordinate(t_vals)).astype(np.int32)
    return np.array([list(zip(x_vals, y_vals))], dtype=np.int32)

# helper function to generate a BGR color for a given radius,
# using the candidate's effective range for normalization.
def get_color_for_radius(r, range_min, range_max):
    # Normalize r between 0 and 1 using its candidate value range.
    normalized = (r - range_min) / (range_max - range_min) if range_max != range_min else 0
    # Avoid red: if normalized is near 0 or near 1, which maps to red, return green.
    if normalized < 0.05 or normalized > 0.95:
        return (0, 255, 0)
    color_rgba = cm.hsv(normalized)
    # Convert RGBA (0-1) to BGR (0-255)
    color_bgr = (int(color_rgba[2]*255), int(color_rgba[1]*255), int(color_rgba[0]*255))
    return color_bgr

# detect hearts using a Hough transform tailored to heart shapes.
# The candidate radii parameter (r_param) can be:
#   - a tuple: (min, max) (continuous range, with 0.5 step),
#   - a list of tuples: [(min1, max1), ...],
#   - or a list of floats: [r1, r2, ...] for discrete candidate radii.
def find_hough_hearts(input_img, edge_img, r_param, vote_threshold):
    img_height, img_width = input_img.shape[:2]
    theta_vals = np.deg2rad(np.arange(0, 360, step=2))
    num_angles = len(theta_vals)

    # Determine candidate type based on r_param
    if isinstance(r_param, tuple):
        candidate_type = "range"
    elif isinstance(r_param, list):
        if isinstance(r_param[0], tuple):
            candidate_type = "range_list"
        else:
            candidate_type = "values"
    else:
        raise ValueError("r_param must be a tuple or a list")

    # Precompute offsets for each theta value.
    offset_x = heart_x_coordinate(theta_vals)
    offset_y = heart_y_coordinate(theta_vals)

    candidate_params = []
    if candidate_type in ["range", "range_list"]:
        # Build candidate parameters from each provided range.
        if candidate_type == "range":
            r_ranges = [r_param]
        else:
            r_ranges = r_param
        for (range_min, range_max) in r_ranges:
            r_vals = np.arange(range_min, range_max, 0.5)
            for r in r_vals:
                for i in range(num_angles):
                    candidate_params.append((r, offset_x[i], offset_y[i], range_min, range_max))
    elif candidate_type == "values":
        candidate_values = r_param
        overall_min = min(candidate_values)
        overall_max = max(candidate_values)
        for r in candidate_values:
            for i in range(num_angles):
                candidate_params.append((r, offset_x[i], offset_y[i], overall_min, overall_max))

    # Accumulate votes for potential heart centers.
    vote_accumulator = defaultdict(int)
    edge_pixels = np.argwhere(edge_img > 0)
    for y_pixel, x_pixel in edge_pixels:
        for candidate in candidate_params:
            r, off_x, off_y, range_min, range_max = candidate
            center_x = int(x_pixel - r * off_x)
            center_y = int(y_pixel - r * off_y)
            if 0 <= center_x < img_width and 0 <= center_y < img_height:
                key = (center_x, center_y, r, range_min, range_max)
                vote_accumulator[key] += 1

    # Filter candidates based on vote ratio.
    detected_candidates = []
    for key, votes in sorted(vote_accumulator.items(), key=lambda item: -item[1]):
        center_x, center_y, r, range_min, range_max = key
        vote_ratio = votes / num_angles
        if vote_ratio > vote_threshold:
            detected_candidates.append((center_x, center_y, r, range_min, range_max, vote_ratio))

    # Remove duplicates: merge detections that are too close.
    pixel_threshold = 10
    filtered_candidates = []
    for candidate in detected_candidates:
        cx, cy, r, rmin, rmax, vr = candidate
        if all((abs(cx - fc[0]) > pixel_threshold or
                abs(cy - fc[1]) > pixel_threshold or
                abs(r - fc[2]) > pixel_threshold)
               for fc in filtered_candidates):
            filtered_candidates.append(candidate)

    # Report the detected heart candidates with their parameters and outline color.
    print(f"Detected {len(filtered_candidates)} hearts using Hough transform (vote_threshold: {vote_threshold}).")
    for cx, cy, r, rmin, rmax, vr in filtered_candidates:
        color = get_color_for_radius(r, rmin, rmax)
        print(f"  Heart candidate - Center: ({cx}, {cy}), Radius: {r}, Vote ratio: {vr:.3f}, Outline color (BGR): {color}")

    # Draw the detected heart contours.
    output_img = input_img.copy()
    for cx, cy, r, rmin, rmax, vr in filtered_candidates:
        heart_contour = generate_heart_contour(scale=r)
        heart_contour_translated = heart_contour + np.array([[cx, cy]])
        contour_color = get_color_for_radius(r, rmin, rmax)
        cv2.drawContours(output_img, [heart_contour_translated], -1, contour_color, 3)

    return output_img

# Define image-specific parameters.
# For "simple" and "med", a continuous range is used.
# For "hard", only two candidate radii are used.
image_parameters = {
    "simple": {"r_range": (5, 10), "vote_threshold": 0.178},
    "med":    {"r_range": (2.5, 10), "vote_threshold": 0.305},
    "hard":   {"r_values": [3.0, 11.0], "vote_threshold": 0.235}
}

# Run detection on sample images.
for image_name in ["simple", "med", "hard"]:
    input_img = cv2.imread(f"{image_name}.jpg")
    if input_img is None:
        print(f"Failed to load {image_name}.jpg")
        continue

    # Preprocess the image.
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
    edge_img = cv2.Canny(blurred_img, 50, 150)

    # Retrieve the specific parameters.
    params = image_parameters[image_name]
    if "r_values" in params:
        r_param = params["r_values"]  # discrete candidate radii for "hard"
    elif "r_range" in params:
        r_param = params["r_range"]
    elif "r_ranges" in params:
        r_param = params["r_ranges"]
    else:
        raise ValueError("No candidate radii parameter found for image")
    vote_threshold = params["vote_threshold"]

    print(f"\nDetecting hearts in '{image_name}' using Hough transform with candidate radii {r_param} and vote threshold {vote_threshold}...")
    result_img = find_hough_hearts(input_img, edge_img, r_param, vote_threshold)

    if result_img is not None:
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected hearts in {image_name}")
        plt.show()
        cv2.imwrite(f"{image_name}_detected.jpg", result_img)
    else:
        print("Detection failed!")

print("Heart detection complete!")
