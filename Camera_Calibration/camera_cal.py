"""
 Camera Calibration example with OpenCV"
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as img

import cv2

def read_data():
    """
        Helper function to Read in Calibration object and image point
    """
    dist_pickle = pickle.load( open( "../Images/calibration_points_pickle.p", "rb" ) )
    obj_points = dist_pickle["objpoints"]
    img_points = dist_pickle["imgpoints"]

    return obj_points, img_points


def cal_undistort(img, obj_points, img_points):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[1::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


if __name__ == "__main__":
    object_points, image_points = read_data()

    # Read in Image
    img = cv2.imread("../Images/Calibration/test_image.jpg")

    undistorted_img = cal_undistort(img, object_points, image_points)

    # Display Results
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Distorted Test Image', fontsize=40)
    ax2.imshow(undistorted_img)
    ax2.set_title('Undistorted Result', fontsize=40)
    plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.0)
    plt.show()

