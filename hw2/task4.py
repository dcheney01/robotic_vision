import numpy as np
import cv2 as cv

data_path = "hw2/data/data_points.txt"

image_points = np.loadtxt(data_path, max_rows=20)
object_points = np.loadtxt(data_path, skiprows=20)

assert image_points.shape == (20, 2)
assert object_points.shape == (20, 3)

intrinsic_params = np.load("hw2/data/intrinsic_params.npy")
distortion_params = np.load("hw2/data/distortion_params.npy")

# estimate object pose
ret, rvec, tvec = cv.solvePnP(object_points, image_points, intrinsic_params, distortion_params)

r_matrix = cv.Rodrigues(rvec)[0]

np.set_printoptions(suppress=True, precision=6)

print("Rotation Matrix: ")
print(r_matrix)

print("Rotation Vector: ")
print(rvec)

print("Translation Vector: ")
print(tvec)




