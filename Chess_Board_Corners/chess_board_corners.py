"""
  Find Chess Board Corners Example
"""

import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

print("Hello, Find Chess Board Corners")

# Read in a calibration image
img = mpimg.imread('../Images/Calibration/GOPR0032.jpg')
plt.imshow(img)
plt.show()

# Map the distorted 2D coordinates (image points) to 3D coordinates (object points)
# Empty arrays to hold points
image_points = []
object_points = []

# Initialize Object Points (0,0,0) ... (7,5,0)
object_coords = np.zeros((8*6,3), np.float32)
object_coords = np.mgrid[0:8, 0:6].T.reshape(-1, 2)   # mgrid returns coordinates for given grid size,
                                                      # re-shape for x,y

# Convert calibration image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Find chessboard corners
ret, corners = cv2.findChessboardCorners(gray_img, (8,6), None)

# if found corners, add them to image points
if ret == True:
    image_points.append(corners)
    object_points.append(object_coords)

    # Display the detected corners
    img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
    plt.imshow(img)
    plt.show()


