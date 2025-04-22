import cv2
import numpy as np
import random

# Load the image
image = cv2.imread("ADE_frame_00000007.jpg")

# Get image dimensions
height, width, _ = image.shape

center = (width // 2, height // 2)
radius = 50


# Create a mask where the circle is white (inpaint area)
mask = np.zeros((height, width), dtype=np.uint8)
cv2.circle(image, center, radius, (255), thickness=-1)

# Inpaint using OpenCV
#inpainted_image = cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

# Save and show results
cv2.imwrite("mask.png", image)
#cv2.imwrite("inpainted.png", inpainted_image)
