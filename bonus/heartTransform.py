# Idan Morad, 316451012
# Nadav Melman, student_id2

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import cm

def parametric_x(t):
	# compute x component for heart shape
	return 14.5 * np.sin(t)**3

def parametric_y(t):
	# compute y component for heart shape
	return 0.5 * np.cos(4*t) + 2 * np.cos(3*t) + 4 * np.cos(2*t) - 13 * np.cos(t)

def generate_heart_contour(scale=1):
	# generate heart contour using parametric eqs
	t_vals = np.linspace(0, 2*np.pi, 360)
	x_vals = (scale * parametric_x(t_vals)).astype(np.int32)
	y_vals = (scale * parametric_y(t_vals)).astype(np.int32)
	return np.array([list(zip(x_vals, y_vals))], dtype=np.int32)

def get_color_for_radius(r, rmin, rmax):
	# get bgr color normalized by radius; avoid red by returning green near boundaries
	norm = (r - rmin) / (rmax - rmin) if rmax != rmin else 0
	if norm < 0.05 or norm > 0.95:
		return (0, 255, 0)
	col_rgba = cm.hsv(norm)
	col_bgr = (int(col_rgba[2]*255), int(col_rgba[1]*255), int(col_rgba[0]*255))
	return col_bgr

def find_hough_shape(image, edge_image, r_min, r_max, bin_threshold):
	# get image dimensions
	img_h, img_w = image.shape[:2]
	# set theta values in degrees
	thetas = np.arange(0, 360, step=2)
	num_thetas = len(thetas)

	# determine candidate radii; for hard image, use discrete values
	if r_min == 3.0 and r_max == 11.0:
		cand_radii = [3.0, 11.0]
	else:
		cand_radii = np.arange(r_min, r_max, 0.5)

	# generate candidate shapes as (r, theta)
	shape_cands = []
	for r in cand_radii:
		for th in thetas:
			shape_cands.append((r, th))

	# initialize hough accumulator
	acc = defaultdict(int)
	# extract edge points from edge_image
	edge_pts = np.argwhere(edge_image > 0)
	for pt in edge_pts:
		y, x = pt
		for (r, th) in shape_cands:
			# compute possible heart center using parametric eqs
			x_center = int(x - r * parametric_x(np.deg2rad(th)))
			y_center = int(y - r * parametric_y(np.deg2rad(th)))
			acc[(x_center, y_center, r)] += 1

	# collect candidate hearts exceeding vote threshold
	out_shapes = []
	for cand, votes in sorted(acc.items(), key=lambda i: -i[1]):
		x_c, y_c, r = cand
		vote_perc = votes / num_thetas
		if vote_perc > bin_threshold:
			out_shapes.append((x_c, y_c, r, vote_perc))

	# remove duplicate detections (using pixel threshold)
	pix_thresh = 10
	post_shapes = []
	for x_c, y_c, r, vp in out_shapes:
		if all((abs(x_c - xs) > pix_thresh or abs(y_c - ys) > pix_thresh or abs(r - rs) > pix_thresh)
			   for xs, ys, rs, _ in post_shapes):
			post_shapes.append((x_c, y_c, r, vp))
	out_shapes = post_shapes

	# set normalization range for color
	norm_min = min(cand_radii) if isinstance(cand_radii, (list, np.ndarray)) else r_min
	norm_max = max(cand_radii) if isinstance(cand_radii, (list, np.ndarray)) else r_max

	# draw detected heart contours (the actual contour, not just the center)
	output_img = image.copy()
	for x_c, y_c, r, vp in out_shapes:
		# generate heart contour and translate to detected center
		heart_cnt = generate_heart_contour(scale=r)
		heart_cnt_trans = heart_cnt + np.array([[x_c, y_c]])
		# get outline color based on detected radius
		out_color = get_color_for_radius(r, norm_min, norm_max)
		cv2.drawContours(output_img, [heart_cnt_trans], -1, out_color, 3)
		# plot contour for visualization
		x1 = heart_cnt_trans[:, 0, 0]
		y1 = heart_cnt_trans[:, 0, 1]
		rgb_color = (out_color[2], out_color[1], out_color[0])
		hex_color = '#%02x%02x%02x' % rgb_color
		plt.plot(x1, y1, markersize=1.5, color=hex_color)
		print(x_c, y_c, r, vp)
	return output_img

# -------------------- main code --------------------

for image_name in ["simple", "med", "hard"]:
	img = cv2.imread(f"{image_name}.jpg")
	if img is None:
		print(f"failed to load {image_name}.jpg")
		continue

	# preprocess image: grayscale, blur, edge detection
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
	edge_img = cv2.Canny(blurred_img, 50, 150)

	# set parameters based on image type
	if image_name == "simple":
		r_min = 5
		r_max = 10
		bin_threshold = 0.178
	elif image_name == "med":
		r_min = 2.5
		r_max = 10
		bin_threshold = 0.305
	elif image_name == "hard":
		r_min = 3.0
		r_max = 11.0
		bin_threshold = 0.24
	else:
		r_min = 1
		r_max = 4
		bin_threshold = 0.1

	print(f"\nattempting to detect hough hearts in '{image_name}'...")
	res_img = find_hough_shape(img, edge_img, r_min, r_max, bin_threshold)
	if res_img is not None:
		plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
		plt.title(f"detected hearts in {image_name}")
		plt.show()
		cv2.imwrite(f"{image_name}_detected.jpg", res_img)
	else:
		print("error in input image!")
	print("hough hearts detection complete!")
