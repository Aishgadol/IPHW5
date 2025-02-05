# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_laplacian_pyramid(image, levels, resize_ratio=0.5):
	# Your code goes here

def restore_from_pyramid(pyramidList, resize_ratio=2):
	# Your code goes here


def validate_operation(img):
	pyr = get_laplacian_pyramid(img, levels)
	img_restored = restore_from_pyramid(pyr)

	plt.title(f"MSE is {np.mean((img_restored - img) ** 2)}")
	plt.imshow(img_restored, cmap='gray')

	plt.show()
	

def blend_pyramids(levels):
	# Your code goes here


apple = cv2.imread('apple.jpg')
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

orange = cv2.imread('orange.jpg')
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

# validate_operation(apple)
# validate_operation(orange)

pyr_apple = get_laplacian_pyramid(apple)
pyr_orange = get_laplacian_pyramid(orange)



pyr_result = []

# Your code goes here

final = restore_from_pyramid(pyr_result)
plt.imshow(final, cmap='gray')
plt.show()

cv2.imwrite("result.jpg", final)

