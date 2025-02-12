import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Heart parametric equation
def parametric_x(t):
    return 14.5 * np.sin(t)**3  # x component

def parametric_y(t):
    return (0.5 * np.cos(4*t) + 2 * np.cos(3*t) + 4 * np.cos(2*t) - 13 * np.cos(t))  # y component

def find_hough_shape(image, edge_image, r_min, r_max, bin_threshold):
    img_height, img_width = image.shape[:2]

    # Define theta and radius ranges
    thetas = np.deg2rad(np.arange(0, 360, step=2))  # Convert to radians
    rs = np.arange(r_min, r_max, 0.5)

    # Precompute values for efficiency
    cos_thetas = parametric_y(thetas)
    sin_thetas = parametric_x(thetas)

    # Generate shape candidates
    shape_candidates = []
    for r in rs:
        for i in range(len(thetas)):
            shape_candidates.append((r, sin_thetas[i], cos_thetas[i]))

    # Hough Accumulator
    accumulator = defaultdict(int)

    # Extract edge points
    edge_points = np.argwhere(edge_image > 0)

    for y, x in edge_points:
        for r, sin_theta, cos_theta in shape_candidates:
            # Calculate possible heart centers
            a = int(x - r * sin_theta)
            b = int(y - r * cos_theta)

            # Ensure within bounds
            if 0 <= a < img_width and 0 <= b < img_height:
                accumulator[(a, b, r)] += 1

    # Output image with detections
    output_img = image.copy()
    out_shapes = []

    # Sort accumulator to find peaks
    num_thetas = len(thetas)
    for candidate_shape, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, r = candidate_shape
        current_vote_percentage = votes / num_thetas

        # Thresholding condition
        if current_vote_percentage > bin_threshold:
            out_shapes.append((x, y, r, current_vote_percentage))

    # Post-processing to remove close detections
    pixel_threshold = 10
    postprocess_shapes = []
    for x, y, r, v in out_shapes:
        if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold and abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_shapes):
            postprocess_shapes.append((x, y, r, v))
    out_shapes = postprocess_shapes

    # Draw detected hearts
    for x, y, r, v in out_shapes:
        t_values = np.linspace(0, 2*np.pi, 200)
        x1 = (x + r * parametric_x(t_values)).astype(int)
        y1 = (y + r * parametric_y(t_values)).astype(int)

        color = (0, 255, 0)  # Green
        for i in range(len(x1) - 1):
            if 0 <= x1[i] < img_width and 0 <= y1[i] < img_height:
                cv2.line(output_img, (x1[i], y1[i]), (x1[i+1], y1[i+1]), color, 1)

        print(f"Detected heart at ({x}, {y}) with radius {r} and confidence {v}")

    return output_img

# Run detection
for IMAGE_NAME in ["simple", "med", "hard"]:
    image = cv2.imread(f'{IMAGE_NAME}.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detector
    edge_image = cv2.Canny(gray, 100, 200)

    r_min, r_max = 10, 100  # Adjust per image
    bin_threshold = 0.1

    if edge_image is not None:
        print(f"Detecting hearts in {IMAGE_NAME}...")
        results_img = find_hough_shape(image, edge_image, r_min, r_max, bin_threshold)

        if results_img is not None:
            plt.imshow(cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB))
            plt.show()
            cv2.imwrite(f'{IMAGE_NAME}_detected.jpg', results_img)
        else:
            print("Detection failed!")

print("Heart detection complete!")
