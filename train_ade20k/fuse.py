import cv2
import numpy as np

# Load the color image
image = cv2.imread("ADE_frame_00000007.jpg")

# Convert to grayscale
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(image, threshold1=100, threshold2=200)

# Convert edges to a 3-channel image
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Blend edges with the original image
fused = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

# Save and display the fused image
cv2.imwrite("fused_canny_color.png", fused)
