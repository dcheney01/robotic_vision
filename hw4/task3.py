import cv2 as cv
import os
import numpy as np


def get_undistorted_imgs(path, intrinsic_params, distortion_params, show=False):
    img = cv.imread(path)
    img_undistorted = cv.undistort(img, intrinsic_params, distortion_params)
    abs_diff = cv.absdiff(img, img_undistorted)

    if show:
        cv.imshow("Original", img)
        cv.imshow("Undistorted", img_undistorted)
        cv.imshow("Abs Diff", abs_diff)
        cv.waitKey(0)

    return img_undistorted, abs_diff

def draw_points_on_img(img, points):
    copy_img = img.copy()
    for point in points:
        cv.circle(copy_img, (point[0], point[1]), 6, (0,0,255), -1)
    return copy_img

def compute_and_draw_epilines(img, points, F):
    for point in points:
        point = np.array([point])
        epilines = cv.computeCorrespondEpilines(point, 1, F)
        epilines = epilines.reshape(-1, 3)
        img = cv.line(img, (0, int(-epilines[0][2] / epilines[0][1])), (img.shape[1], int((-epilines[0][2] - epilines[0][0]*img.shape[1]) / epilines[0][1])), (0, 255, 0), 2)
    return img

if __name__== "__main__":
    left_image_path = "hw4/data/Stereo/L/0.png"
    right_image_path = "hw4/data/Stereo/R/0.png"

    left_intrinsic_matrix = np.load("hw4/data/left_camera_intrinsic_params.npy")
    left_distortion_matrix = np.load("hw4/data/left_camera_distortion_params.npy")
    right_intrinsic_matrix = np.load("hw4/data/right_camera_intrinsic_params.npy")
    right_distortion_matrix = np.load("hw4/data/right_camera_distortion_params.npy")
    E = np.load("hw4/data/stereo_E.npy")
    F = np.load("hw4/data/stereo_F.npy")
    print(F)

    left_img_undistorted, left_abs_diff = get_undistorted_imgs(left_image_path, left_intrinsic_matrix, left_distortion_matrix, show=False)
    right_img_undistorted, right_abs_diff = get_undistorted_imgs(right_image_path, right_intrinsic_matrix, right_distortion_matrix, show=False)

    left_points = np.array([[336, 163], 
                            [362, 233], 
                            [461, 119]])
    right_points = np.array([[350, 371],
                             [399, 178],
                             [489, 233]])
    
    left_points_img = draw_points_on_img(left_img_undistorted, left_points)
    right_points_img = draw_points_on_img(right_img_undistorted, right_points)

    left_epiline_img = compute_and_draw_epilines(left_points_img, right_points, F)
    right_epiline_img = compute_and_draw_epilines(right_points_img, left_points, F)

    cv.imshow("Left Epiline", left_epiline_img)
    cv.imshow("Right Epiline", right_epiline_img)
    cv.waitKey(0)    

