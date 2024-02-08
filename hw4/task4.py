import numpy as np
import cv2 as cv
import os

if __name__== "__main__":
    left_intrinsic_matrix = np.load("hw4/data/left_camera_intrinsic_params.npy")
    left_distortion_matrix = np.load("hw4/data/left_camera_distortion_params.npy")
    right_intrinsic_matrix = np.load("hw4/data/right_camera_intrinsic_params.npy")
    right_distortion_matrix = np.load("hw4/data/right_camera_distortion_params.npy")
    E = np.load("hw4/data/stereo_E.npy")
    F = np.load("hw4/data/stereo_F.npy")
    R = np.load("hw4/data/stereo_R.npy")
    T = np.load("hw4/data/stereo_T.npy")

    # use stereo rectify to get R1, R2, P1, P2, Q
    size = (640, 480)
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(left_intrinsic_matrix, left_distortion_matrix, right_intrinsic_matrix, right_distortion_matrix, size, R, T)

    # use initUndistortRectifyMap to get map1, map2
    left_map1, left_map2 = cv.initUndistortRectifyMap(left_intrinsic_matrix, left_distortion_matrix, R1, P1, size, cv.CV_32FC1)
    right_map1, right_map2 = cv.initUndistortRectifyMap(right_intrinsic_matrix, right_distortion_matrix, R2, P2, size, cv.CV_32FC1)

    # use remap to get rectified images
    left_image_path = "hw4/data/Stereo/L/0.png"
    right_image_path = "hw4/data/Stereo/R/0.png"

    left_img = cv.imread(left_image_path)
    right_img = cv.imread(right_image_path)

    left_rectified_img = cv.remap(left_img, left_map1, left_map2, cv.INTER_LINEAR)
    right_rectified_img = cv.remap(right_img, right_map1, right_map2, cv.INTER_LINEAR)

    abs_diff_left = cv.absdiff(left_img, left_rectified_img)
    abs_diff_right = cv.absdiff(right_img, right_rectified_img)

    # draw a few horizontal lines to show rectification
    for i in range(0, left_rectified_img.shape[0], 35):
        cv.line(left_rectified_img, (0, i), (left_rectified_img.shape[1], i), (0, 255, 0), 1)
        cv.line(right_rectified_img, (0, i), (right_rectified_img.shape[1], i), (0, 255, 0), 1)

    cv.imshow("Left Rectified", left_rectified_img)
    cv.imshow("Right Rectified", right_rectified_img)

    cv.imshow("Left Abs Diff", abs_diff_left)
    cv.imshow("Right Abs Diff", abs_diff_right)

    cv.imshow("Left", left_img)
    cv.imshow("Right", right_img)
    cv.waitKey(0)

    if True:
        cv.imwrite("hw4/output_imgs/left_rectified.jpg", left_rectified_img)
        cv.imwrite("hw4/output_imgs/right_rectified.jpg", right_rectified_img)
        cv.imwrite("hw4/output_imgs/left_abs_diff.jpg", abs_diff_left)
        cv.imwrite("hw4/output_imgs/right_abs_diff.jpg", abs_diff_right)
        cv.imwrite("hw4/output_imgs/left.jpg", left_img)
        cv.imwrite("hw4/output_imgs/right.jpg", right_img)




