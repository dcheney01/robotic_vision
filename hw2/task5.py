import cv2 as cv
import os
import numpy as np

from task2 import detect_corners

def capture_imgs(data_path, num_imgs):
    cap = cv.VideoCapture(1)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    for i in range(num_imgs):
        ret, frame = cap.read()
        ret, frame = cap.read()

        cv.imshow("Capture", frame)

        # wait until key is pressed and if the key is spacebar, save
        key = cv.waitKey(0)
        if key == 32:
            img_path = os.path.join(data_path, "img" + str(i) + ".jpg")
            cv.imwrite(img_path, frame)
            print("Saved image to: ", img_path)
        elif key == 27:
            continue
 
    cap.release()



if __name__ == "__main__":
    data_path = "hw2/data/cal_imgs_webcam"
    num_imgs = 40
    size = (9, 7)

    # capture_imgs(data_path, num_imgs)
    
    objp = np.zeros((size[0]*size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    data = sorted(os.listdir(data_path))
    for img in data:
        img_path = os.path.join(data_path, img)
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

    np.set_printoptions(suppress=True, precision=6)

    print("Focal Length (x): ", focal_length_x_mm, "mm")
    print("Focal Length (y): ", focal_length_y_mm, "mm")
    print("Focal Length (x): ", focal_length_x_pixels, "pixels")
    print("Focal Length (y): ", focal_length_y_pixels, "pixels")

    print("Intrinsic Matrix: ")
    print(mtx)

    print("Distortion Coefficients: ")
    print(dist)

    np.save("hw2/data/intrinsic_params_webcam.npy", mtx)
    np.save("hw2/data/distortion_params_webcam.npy", dist)



