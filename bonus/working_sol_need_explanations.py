import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Parametric equations for a heart shape
def parametric_x(t):
    return 14.5 * np.sin(t) ** 3  # x component

def parametric_y(t):
    return (
            0.5 * np.cos(4 * t) + 2 * np.cos(3 * t) + 4 * np.cos(2 * t) - 13 * np.cos(t)
    )  # y component

def generate_heart_contour(scale=1):
    #generates a generic heart shape as a reference contour for shape matching.
    t_values = np.linspace(0, 2 * np.pi, 300)
    x = (scale * parametric_x(t_values)).astype(np.int32)
    y = (scale * parametric_y(t_values)).astype(np.int32)
    return np.array([list(zip(x, y))], dtype=np.int32)

def find_hough_shape(image, edge_image, r_min, r_max, bin_threshold):
    img_height, img_width = image.shape[:2]

    # define theta and radius ranges
    thetas = np.deg2rad(np.arange(0, 360, step=2))  # Convert to radians
    rs = np.arange(r_min, r_max, 0.5)

    # precompute values for efficiency
    cos_thetas = parametric_y(thetas)
    sin_thetas = parametric_x(thetas)

    # Generate shape candidates
    shape_candidates = [
        (r, sin_thetas[i], cos_thetas[i]) for r in rs for i in range(len(thetas))
    ]

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
        if all(
                abs(x - xc) > pixel_threshold
                or abs(y - yc) > pixel_threshold
                and abs(r - rc) > pixel_threshold
                for xc, yc, rc, v in postprocess_shapes
        ):
            postprocess_shapes.append((x, y, r, v))
    out_shapes = postprocess_shapes

    # Use contours for shape matching
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reference_heart = generate_heart_contour()

    # Filter for only heart-like shapes
    best_matches = []
    for contour in contours:
        match_score = cv2.matchShapes(contour, reference_heart, cv2.CONTOURS_MATCH_I1, 0)
        if match_score < 0.2:  # Strict threshold for heart detection
            best_matches.append(contour)

    # Draw only detected heart outlines
    cv2.drawContours(output_img, best_matches, -1, (255,0 , 0), 3)
    print(f"Detected {len(best_matches)} hearts.")

    return output_img

# Run detection
for IMAGE_NAME in ["simple", "med", "hard"]:
    image = cv2.imread(f"{IMAGE_NAME}.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detector with Gaussian Blur (as per Hough method in slides)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edge_image = cv2.Canny(blurred, 50, 150)  # Hough method recommendation

    r_min, r_max = 10, 100  # Adjust per image
    bin_threshold = 0.1

    if edge_image is not None:
        print(f"Detecting hearts in {IMAGE_NAME}...")
        results_img = find_hough_shape(image, edge_image, r_min, r_max, bin_threshold)

        if results_img is not None:
            plt.imshow(cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB))
            plt.show()
            cv2.imwrite(f"{IMAGE_NAME}_detected.jpg", results_img)
        else:
            print("Detection failed!")

print("Heart detection complete!")
