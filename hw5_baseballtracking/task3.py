import numpy as np
import cv2 as cv
import os

def find_contours(img, prev_image):
    abs_diff = cv.absdiff(prev_image, img)
    gray = cv.cvtColor(abs_diff, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    edges = cv.Canny(blurred, 25, 50)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def get_roi_box(contours):
    aspect_ratios = [cv.boundingRect(c)[2] / cv.boundingRect(c)[3] for c in contours]
    max_contour = contours[np.argmin(np.abs(np.array(aspect_ratios) - 1))]
    x, y, w, h = cv.boundingRect(max_contour)
    return x-15, y-15, 50, 50

def find_ball_center(img, prev_img):
    contours = find_contours(img, prev_img)

    if len(contours) > 0:
        x, y, w, h = get_roi_box(contours)
        
        max_y = y+h if y+h < img.shape[0] else img.shape[0]
        max_x = x+w if x+w < img.shape[1] else img.shape[1]

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_roi_gray = gray[y:max_y, x:max_x]
        circles = cv.HoughCircles(img_roi_gray, cv.HOUGH_GRADIENT, 4.89, 10000, param1=250, param2=15, minRadius=1, maxRadius=13)

        if circles is not None:
            circles = np.uint16(np.around(circles)).squeeze()
            circles[0] += x
            circles[1] += y
            return circles
    return None

def get_undistorted_points(left_points, right_points, left_intrinsic_params, left_distortion_params, right_intrinsic_params, right_distortion_params, R, T, F):
    RLeft, RRight, PLeft, PRight, Q, _, _ = cv.stereoRectify(left_intrinsic_params, left_distortion_params, right_intrinsic_params, right_distortion_params, (640, 480), R, T)
    
    left_undistorted_points = cv.undistortPoints(left_points, left_intrinsic_params, left_distortion_params, R=RLeft, P=PLeft)
    right_undistorted_points = cv.undistortPoints(right_points, right_intrinsic_params, right_distortion_params, R=RRight, P=PRight)

    return left_undistorted_points, right_undistorted_points, Q, PLeft, PRight

def get_3D_points(left_undistorted_points, right_undistorted_points, Q):
    left_undistorted_points = left_undistorted_points[:,0,:]
    right_undistorted_points = right_undistorted_points[:,0,:]

    disparity = (left_undistorted_points[:,0] - right_undistorted_points[:,0]).reshape(4,1)

    left_3D_undistorted_points = np.hstack((left_undistorted_points, disparity)).reshape(4,1,3)
    right_3D_undistorted_points = np.hstack((right_undistorted_points, disparity)).reshape(4,1,3)

    left_3D_points_warped = cv.perspectiveTransform(left_3D_undistorted_points, Q)
    right_3D_points_warped = cv.perspectiveTransform(right_3D_undistorted_points, Q)

    return left_3D_points_warped, right_3D_points_warped

if __name__=="__main__":
    img_folder = "robotic_vision/hw5_baseballtracking/data/20240215113225/"

    left_img_folder = img_folder + "L/"
    right_img_folder = img_folder + "R/"

    left_img_paths = sorted(os.listdir(left_img_folder), key=lambda x: int(x.split(".")[0]))
    right_img_paths = sorted(os.listdir(right_img_folder), key=lambda x: int(x.split(".")[0]))

    prev_left_img = cv.imread(left_img_folder + left_img_paths[0])
    prev_right_img = cv.imread(right_img_folder + right_img_paths[0])

    for i in range(1, len(left_img_paths)):
        left_img = cv.imread(left_img_folder + left_img_paths[i])
        right_img = cv.imread(right_img_folder + right_img_paths[i])

        # Find points of ball in left and right images
        left_ball_center = find_ball_center(left_img, prev_left_img)
        right_ball_center = find_ball_center(right_img, prev_right_img)

        # Find 3D ball location w.r.t. the left camera
        left_intrinsic_matrix = np.load("hw4/data/left_camera_intrinsic_params.npy")
        left_distortion_matrix = np.load("hw4/data/left_camera_distortion_params.npy")
        right_intrinsic_matrix = np.load("hw4/data/right_camera_intrinsic_params.npy")
        right_distortion_matrix = np.load("hw4/data/right_camera_distortion_params.npy")

        E = np.load("hw4/data/stereo_E.npy")
        F = np.load("hw4/data/stereo_F.npy")
        R = np.load("hw4/data/stereo_R.npy")
        T = np.load("hw4/data/stereo_T.npy")

        left_undistorted_points, right_undistorted_points, Q, PLeft, PRight = get_undistorted_points(left_ball_center, right_ball_center, left_intrinsic_matrix, left_distortion_matrix, right_intrinsic_matrix, right_distortion_matrix, R, T, F)
        left_3D_points_warped, right_3D_points_warped = get_3D_points(left_undistorted_points, right_undistorted_points, Q)

        # Transfer points to catcher's coordinate system
        

        # Plot one graph of the ball location with the Y and Z coordinates

        # Plot another graph of the ball location with the X and Z coordinates