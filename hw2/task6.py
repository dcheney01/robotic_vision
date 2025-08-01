import cv2 as cv
import numpy as np

img_path = "hw2/data/cal_imgs_webcam/img3.jpg"

intrinsic_params = np.load("hw2/data/intrinsic_params_webcam.npy")
distortion_params = np.load("hw2/data/distortion_params_webcam.npy")

img = cv.imread(img_path)
undistorted_img = cv.undistort(img, intrinsic_params, distortion_params)

abs_diff_img = cv.absdiff(img, undistorted_img)

cv.imshow("Original", img)
cv.imshow("Undistorted", undistorted_img)
cv.imshow("Abs Diff", abs_diff_img)

cv.imwrite("hw2/output_imgs/abs_diff_webcam.jpg", abs_diff_img)

cv.waitKey(0)