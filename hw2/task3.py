import numpy as np
import cv2 as cv2

far_img_path = "hw2/data/Far.jpg"
close_img_path = "hw2/data/Close.jpg"
turn_img_path = "hw2/data/Turn.jpg"

intrinsic_params = np.load("hw2/data/intrinsic_params.npy")
distortion_params = np.load("hw2/data/distortion_params.npy")

# undistort images
far_img = cv2.imread(far_img_path)
close_img = cv2.imread(close_img_path)
turn_img = cv2.imread(turn_img_path)

undistorted_far_img = cv2.undistort(far_img, intrinsic_params, distortion_params)
undistorted_close_img = cv2.undistort(close_img, intrinsic_params, distortion_params)
undistorted_turn_img = cv2.undistort(turn_img, intrinsic_params, distortion_params)

abs_diff_far_img = cv2.absdiff(far_img, undistorted_far_img)
abs_diff_close_img = cv2.absdiff(close_img, undistorted_close_img)
abs_diff_turn_img = cv2.absdiff(turn_img, undistorted_turn_img)

cv2.imwrite("hw2/output_imgs/far_abs_diff.jpg", abs_diff_far_img)
cv2.imwrite("hw2/output_imgs/close_abs_diff.jpg", abs_diff_close_img)
cv2.imwrite("hw2/output_imgs/turn_abs_diff.jpg", abs_diff_turn_img)

cv2.imshow("Far", far_img)
cv2.imshow("Close", close_img)
cv2.imshow("Turn", turn_img)

cv2.imshow("Far Undistorted", undistorted_far_img)
cv2.imshow("Close Undistorted", undistorted_close_img)
cv2.imshow("Turn Undistorted", undistorted_turn_img)

cv2.imshow("Far Abs Diff", abs_diff_far_img)
cv2.imshow("Close Abs Diff", abs_diff_close_img)
cv2.imshow("Turn Abs Diff", abs_diff_turn_img)

cv2.waitKey(0)
