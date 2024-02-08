import numpy as np
import cv2 as cv
import os

def get_img_points(data_path, size=(10,7)):
    img_points = []

    for img in os.listdir(data_path):
        img_path = os.path.join(data_path, img)
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, size, None)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_points.append(corners2)
        else:
            print(f"No corners detected in {img_path}")
    return img_points


def calibrate_stereo_camera(left_intrinsic_matrix, left_distortion_matrix, right_intrinsic_matrix, right_distortion_matrix, stereo_img_path):
    size = (10,7)
    objp = np.zeros((size[0]*size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1,2) * 3.88
    objpoints = [objp]*len(os.listdir(stereo_img_path+"/L"))

    left_imgpoints = get_img_points(stereo_img_path+"/L")
    right_imgpoints = get_img_points(stereo_img_path+"/R")

    ret, _, _, _, _, R, T, E, F = cv.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints, left_intrinsic_matrix, \
                                                     left_distortion_matrix, right_intrinsic_matrix, right_distortion_matrix, (640,480), \
                                                     criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001), flags=cv.CALIB_FIX_INTRINSIC)

    np.set_printoptions(suppress=True, precision=4)
    print("Rotation Matrix: ")
    print(R)
    print("Translation Matrix: ")
    print(T)
    print("Essential Matrix: ")
    print(E)
    print("Fundamental Matrix: ")
    print(F)

    np.save("hw4/data/stereo_R.npy", R)
    np.save("hw4/data/stereo_T.npy", T)
    np.save("hw4/data/stereo_E.npy", E)
    np.save("hw4/data/stereo_F.npy", F)

    return R, T, E, F

if __name__ == "__main__":
    left_intrinsic_matrix = np.load("hw4/data/left_camera_intrinsic_params.npy")
    left_distortion_matrix = np.load("hw4/data/left_camera_distortion_params.npy")
    right_intrinsic_matrix = np.load("hw4/data/right_camera_intrinsic_params.npy")
    right_distortion_matrix = np.load("hw4/data/right_camera_distortion_params.npy")
    stereo_img_path = "hw4/data/Stereo"

    calibrate_stereo_camera(left_intrinsic_matrix, left_distortion_matrix, right_intrinsic_matrix, right_distortion_matrix, stereo_img_path)