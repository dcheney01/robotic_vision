import numpy as np
import cv2 as cv
import os

def get_img_points(data_path, size=(10,7)):
    img_points = []

    for img in sorted(os.listdir(data_path)):
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

    np.set_printoptions(suppress=True, precision=6)
    print("Rotation Matrix: ")
    print(R)
    print("Translation Matrix: ")
    print(T)
    print("Essential Matrix: ")
    print(E)
    print("Fundamental Matrix: ")
    print(F)

    # np.save("hw4/data/stereo_R.npy", R)
    # np.save("hw4/data/stereo_T.npy", T)
    # np.save("hw4/data/stereo_E.npy", E)
    # np.save("hw4/data/stereo_F.npy", F)

    return R, T, E, F

if __name__ == "__main__":
    left_intrinsic_matrix = np.load("hw4/data/left_camera_intrinsic_params.npy")
    left_distortion_matrix = np.load("hw4/data/left_camera_distortion_params.npy")
    right_intrinsic_matrix = np.load("hw4/data/right_camera_intrinsic_params.npy")
    right_distortion_matrix = np.load("hw4/data/right_camera_distortion_params.npy")
    stereo_img_path = "/home/daniel/software/robotic_vision/hw4/data/Stereo/"#Practice/stereo"

    calibrate_stereo_camera(left_intrinsic_matrix, left_distortion_matrix, right_intrinsic_matrix, right_distortion_matrix, stereo_img_path)


    # stereo_img_path = "/home/daniel/software/robotic_vision/hw4/data/Practice/stereo"

    # left_camera_test_intrinsics = np.array([[1153.7918353122, 0.0000000000, 311.6495082713],
    #                                        [0.0000000000, 1152.7061368496, 247.7409370695],
    #                                        [0.0000000000, 0.0000000000, 1.0000000000]])
    # left_camera_test_distortion = np.array([-0.2574338164, 0.3395576609, 0.0011179409, -0.0002030712, -0.5947353243])
    # right_camera_test_intrinsics = np.array([[1149.6505965772, 0.0000000000, 326.3569432986],
    #                                          [0.0000000000, 1148.0218738819, 224.6062742604],
    #                                          [0.0000000000, 0.0000000000, 1.0000000000]])
    # right_camera_test_distortion = np.array([-0.2950621013, 1.1296741454, -0.0010482716, -0.0014052463, -9.9589684633])

    # calibrate_stereo_camera(left_camera_test_intrinsics, left_camera_test_distortion, right_camera_test_intrinsics, right_camera_test_distortion, stereo_img_path)