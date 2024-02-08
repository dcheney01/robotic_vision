import cv2 as cv
import os
import numpy as np

def detect_corners(path, size, display=False):
    img = cv.imread(path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray_img, size, None)
    if ret:
        refined_corners = cv.cornerSubPix(gray_img, corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        if display:
            refined_corners_img = img.copy()
            cv.drawChessboardCorners(refined_corners_img, size, refined_corners, ret)

            cv.imshow("Refined Corners", refined_corners_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return refined_corners
    else:
        print(f"No corners detected in {path}")


def calibrate_camera(data_name, data_folder):
    size = (10,7)
    objp = np.zeros((size[0]*size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1,2)
    objpoints = []
    imgpoints = []

    data = sorted(os.listdir(data_folder))
    for img in data:
        img_path = os.path.join(data_folder, img)
        corners = detect_corners(img_path, size, display=False)

        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (640,480), None, None)

    pixels_size = 7.4e-3
    pixels_per_mm = 1 / pixels_size
    focal_length_x_pixels = mtx[0][0]
    focal_length_y_pixels = mtx[1][1]

    focal_length_x_mm = focal_length_x_pixels / pixels_per_mm
    focal_length_y_mm = focal_length_y_pixels / pixels_per_mm

    np.set_printoptions(suppress=True, precision=4)

    print(f"{data_name} Focal Length (x): ", focal_length_x_mm, "mm")
    print(f"{data_name} Focal Length (y): ", focal_length_y_mm, "mm")
    print(f"{data_name} Average Focal Length: ", (focal_length_x_mm + focal_length_y_mm) / 2, "mm")

    print("Instrinsic Matrix: ")
    print(mtx)

    print("Distortion Coefficients: ")
    print(dist)

    np.save(f"hw4/data/{data_name}_intrinsic_params.npy", mtx)
    np.save(f"hw4/data/{data_name}_distortion_params.npy", dist)

if __name__ == "__main__":
    # img_path = "hw2/data/cal_imgs/AR1.jpg"


    data_name = "left_camera"
    data_folder = "hw4/data/LeftOnly/L"
    calibrate_camera(data_name, data_folder)
    print()

    data_name = "right_camera"
    data_folder = "hw4/data/RightOnly/R"
    calibrate_camera(data_name, data_folder)
    