import cv2 as cv
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

def get_undistorted_points(left_points, right_points, left_intrinsic_params, left_distortion_params, right_intrinsic_params, right_distortion_params, R, T, F):
    RLeft, RRight, PLeft, PRight, Q, _, _ = cv.stereoRectify(left_intrinsic_params, left_distortion_params, right_intrinsic_params, right_distortion_params, (640, 480), R, T)
    
    left_undistorted_points = cv.undistortPoints(left_points, left_intrinsic_params, left_distortion_params, R=RLeft, P=PLeft)
    right_undistorted_points = cv.undistortPoints(right_points, right_intrinsic_params, right_distortion_params, R=RRight, P=PRight)

    return left_undistorted_points, right_undistorted_points, Q, PLeft, PRight

def draw_points_on_img(img, points, points_3D):
    copy_img = img.copy()
    points = points[:,0,:]
    points_3D = points_3D[:,0,:]
    for i in range(len(points)):
        cv.circle(copy_img, (int(points[i][0]), int(points[i][1])), 6, (0,0,255), -1)
        # write the 3d point at the same spot for each corner
        cv.putText(copy_img, f"({points_3D[i][0]:.2f}, {points_3D[i][1]:.2f}, {points_3D[i][2]:.2f})", (int(points[i][0] - 100), int(points[i][1]-10)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
    return copy_img

def get_3D_points(left_undistorted_points, right_undistorted_points, Q):
    left_undistorted_points = left_undistorted_points[:,0,:]
    right_undistorted_points = right_undistorted_points[:,0,:]

    disparity = (left_undistorted_points[:,0] - right_undistorted_points[:,0]).reshape(4,1)

    left_3D_undistorted_points = np.hstack((left_undistorted_points, disparity)).reshape(4,1,3)
    right_3D_undistorted_points = np.hstack((right_undistorted_points, disparity)).reshape(4,1,3)

    left_3D_points_warped = cv.perspectiveTransform(left_3D_undistorted_points, Q)
    right_3D_points_warped = cv.perspectiveTransform(right_3D_undistorted_points, Q)

    return left_3D_points_warped, right_3D_points_warped

if __name__ == "__main__":
    image_num = "0"
    chessboard_img_path = "hw4/data/Stereo/"
    left_path = chessboard_img_path + "L/" + image_num + ".png"
    right_path = chessboard_img_path + "R/" + image_num + ".png"

    left_intrinsic_matrix = np.load("hw4/data/left_camera_intrinsic_params.npy")
    left_distortion_matrix = np.load("hw4/data/left_camera_distortion_params.npy")
    right_intrinsic_matrix = np.load("hw4/data/right_camera_intrinsic_params.npy")
    right_distortion_matrix = np.load("hw4/data/right_camera_distortion_params.npy")

    E = np.load("hw4/data/stereo_E.npy")
    F = np.load("hw4/data/stereo_F.npy")
    R = np.load("hw4/data/stereo_R.npy")
    T = np.load("hw4/data/stereo_T.npy")

    np.set_printoptions(suppress=True, precision=6)

    left_corners = detect_corners(left_path, (10,7), display=False)
    right_corners = detect_corners(right_path, (10,7), display=False)

    left_outer_four_corners = np.array([left_corners[0], left_corners[9], left_corners[60], left_corners[69]])
    right_outer_four_corners = np.array([right_corners[0], right_corners[9], right_corners[60], right_corners[69]])
    
    left_undistorted_points, right_undistorted_points, Q, PLeft, PRight = get_undistorted_points(left_outer_four_corners, right_outer_four_corners, left_intrinsic_matrix, left_distortion_matrix, right_intrinsic_matrix, right_distortion_matrix, R, T, F)

    left_3D_points_warped, right_3D_points_warped = get_3D_points(left_undistorted_points, right_undistorted_points, Q)
    print(f"Left 3D Points:\n {left_3D_points_warped}\n")
    print(f"Right 3D Points:\n {right_3D_points_warped}")

    # draw points on the image with 3D coordinates
    left_img = cv.imread(left_path)
    right_img = cv.imread(right_path)

    left_img_with_points = draw_points_on_img(left_img, left_outer_four_corners, left_3D_points_warped)
    right_img_with_points = draw_points_on_img(right_img, right_outer_four_corners, right_3D_points_warped)

    # cv.imshow("Left Image with Points", left_img_with_points)
    # cv.imshow("Right Image with Points", right_img_with_points)
    # cv.waitKey(0)

    # cv.imwrite("hw5_baseballtracking/output_imgs/task1_LPoints.jpg", left_img_with_points)
    # cv.imwrite("hw5_baseballtracking/output_imgs/task1_RPoints.jpg", right_img_with_points)

    right_3D_points_warped = right_3D_points_warped.reshape(4,3)
    left_3D_points_warped = left_3D_points_warped.reshape(4,3)

    # Check that the 3D points make sense
    left_img_with_points_from_right = right_3D_points_warped + np.array([[np.linalg.norm(T), 0, 0]])
    right_img_with_points_from_left = left_3D_points_warped - np.array([[np.linalg.norm(T), 0, 0]])

    print(f"Left 3D Points from Right:\n {left_img_with_points_from_right}\n")
    print(f"Right 3D Points from Left:\n {right_img_with_points_from_left}")

    print(f"Diff between left_3D_points_warped and left_img_with_points_from_right: {left_3D_points_warped - left_img_with_points_from_right}")
    print(f"Diff between right_3D_points_warped and right_img_with_points_from_left: {right_3D_points_warped - right_img_with_points_from_left}")




