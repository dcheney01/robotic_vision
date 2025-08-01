import numpy as np
import cv2 as cv
import cv2
import os

def find_contours(img, prev_image):
    abs_diff = cv2.absdiff(prev_image, img)
    gray = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 25, 50)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow("Abs Diff", abs_diff)
    # cv2.imshow("Blurred", blurred)
    # cv2.imshow("Edges", edges)
    return contours

def get_roi_box(contours):
    aspect_ratios = [cv2.boundingRect(c)[2] / cv2.boundingRect(c)[3] for c in contours]
    max_contour = contours[np.argmin(np.abs(np.array(aspect_ratios) - 1))]
    x, y, w, h = cv2.boundingRect(max_contour)
    return x-30, y-30, 75, 75

def find_ball_center(img, prev_image):

    contours = find_contours(img, prev_image)

    if len(contours) > 0:
        x, y, w, h = get_roi_box(contours)

        max_y = y+h if y+h < img.shape[0] else img.shape[0]
        max_x = x+w if x+w < img.shape[1] else img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_roi_gray = gray[y:max_y, x:max_x]
        circles = cv2.HoughCircles(img_roi_gray, cv2.HOUGH_GRADIENT, 4.89, 10000, param1=200, param2=15, minRadius=1, maxRadius=16)
    
        if circles is not None:
        #     cv.imshow("Gray", gray)
        #     cv.imshow("ROI", img_roi_gray)
        #     circles_img = img.copy()
            # cv.waitKey(0)
            # # draw circles on circles_img   
            # cv.circle(circles_img, (x+int(circles[0][0][0]), y+int(circles[0][0][1])), int(circles[0][0][2]), (0, 255, 0), 2)
            # cv.circle(circles_img, (x+int(circles[0][0][0]), y+int(circles[0][0][1])), 1, (0, 0, 255), -1)
            # # print(f"Radius: {circles[0][0][2]}")
            # cv.imshow("Circles", circles_img)
            # cv.waitKey(0)

            circles = np.uint16(np.around(circles)).squeeze()
            circles[0] += x
            circles[1] += y
            # roi = [circles[0]-3*circles[2], circles[1]-3*circles[2], circles[2]*8, circles[2]*8]
            # roi[0] = 0 if roi[0] < 0 else roi[0]
            # roi[1] = 0 if roi[1] < 0 else roi[1]
            # print(roi)
            return circles[:2]
        else: 
            print("No circles found")
    else:
        print("Didn't find any contours")
    # x, y, w, h = roi
    # max_y = y+h if y+h < img.shape[0] else img.shape[0]
    # max_x = x+w if x+w < img.shape[1] else img.shape[1]

    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img_roi_gray = gray[y:max_y, x:max_x]
    # img_roi_gray_blurred = cv.GaussianBlur(img_roi_gray, (5, 5), 0)
    # circles = cv.HoughCircles(img_roi_gray_blurred, cv.HOUGH_GRADIENT, 4.89, 1000, param1=120, param2=21, minRadius=3, maxRadius=16)

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
    img_folder = "hw5_baseballtracking/data/20240215113225/"
    # img_folder = "hw5_baseballtracking/data/20240215112959/"

    left_img_folder = img_folder + "L/"
    right_img_folder = img_folder + "R/"
    left_img_paths = sorted(os.listdir(left_img_folder), key=lambda x: int(x.split(".")[0]))
    right_img_paths = sorted(os.listdir(right_img_folder), key=lambda x: int(x.split(".")[0]))

    left_intrinsic_matrix = np.load("hw4/data/left_camera_intrinsic_params.npy")
    left_distortion_matrix = np.load("hw4/data/left_camera_distortion_params.npy")
    right_intrinsic_matrix = np.load("hw4/data/right_camera_intrinsic_params.npy")
    right_distortion_matrix = np.load("hw4/data/right_camera_distortion_params.npy")

    E = np.load("hw4/data/stereo_E.npy")
    F = np.load("hw4/data/stereo_F.npy")
    R = np.load("hw4/data/stereo_R.npy")
    T = np.load("hw4/data/stereo_T.npy")

    left_roi = [340, 75, 50, 60]
    right_roi = [260, 75, 50, 60]

    prev_left_img = cv.imread(left_img_folder + left_img_paths[0])
    prev_right_img = cv.imread(right_img_folder + right_img_paths[0])

    all_left_points = []
    # all_right_points = []

    for i in range(1, len(left_img_paths)):
        # print(f"\n\nProcessing: {left_img_paths[i]}, {i}")
        left_img = cv.imread(left_img_folder + left_img_paths[i])
        right_img = cv.imread(right_img_folder + right_img_paths[i])

        # Find points of ball in left and right images
        left_ball_center = find_ball_center(left_img, prev_left_img)
        right_ball_center = find_ball_center(right_img, prev_right_img)

        if left_ball_center is not None and right_ball_center is not None:
            # Repeat the points 4 times
            left_ball_center_points = np.repeat(left_ball_center.reshape(1,1,2), 4, axis=0).astype(np.float32)
            right_ball_center_points = np.repeat(right_ball_center.reshape(1,1,2), 4, axis=0).astype(np.float32)
            # Find 3D ball location w.r.t. the left camera
            left_undistorted_points, right_undistorted_points, Q, PLeft, PRight = get_undistorted_points(left_ball_center_points, right_ball_center_points, left_intrinsic_matrix, left_distortion_matrix, right_intrinsic_matrix, right_distortion_matrix, R, T, F)
            left_3D_points_warped, right_3D_points_warped = get_3D_points(left_undistorted_points, right_undistorted_points, Q)

            # Transfer points to catcher's coordinate system
            left_3D_points_catcher = left_3D_points_warped[0] - np.array([[np.linalg.norm(T)/2, 0, 0]])

            all_left_points.append(left_3D_points_catcher.squeeze())

        prev_left_img = left_img
        prev_right_img = right_img



    points_array = np.array(all_left_points)
    print(len(all_left_points))
    # Fit a best fit line to the yz and xz trajectories with parabola
    yz_model = np.polyfit(points_array[:, 2], points_array[:, 1], 2)
    xz_model = np.polyfit(points_array[:, 2], points_array[:, 0], 1) 

    #print intercepts to get x,y point when z=0
    print(f"YZ Intercept: {yz_model[2]}")
    print(f"XZ Intercept: {xz_model[1]}")



    import matplotlib.pyplot as plt

    left_camera_location = np.array([-10, 0, 0])
    catcher_location = np.array([0, 0, 0])
    right_camera_location = np.array([10, 0, 0])

    # two subplots, one has y, z trajectory, the other has x, z trajectory
    fig, axs = plt.subplots(2)
    # make the 0 for the x aaxis on the left
    fig.suptitle('3D Ball Location')
    axs[0].scatter(points_array[:,2], points_array[:,1], c='g')
    axs[0].plot(left_camera_location[2], left_camera_location[1], 'ro')
    axs[0].plot(catcher_location[2], catcher_location[1], 'ko')
    axs[0].plot(right_camera_location[2], right_camera_location[1], 'bo')
    # plot the yz model
    z = np.linspace(max(points_array[:,2])+30, 0, 100)
    y = yz_model[0]*z**2 + yz_model[1]*z + yz_model[2]
    axs[0].plot(z, y, 'r')
    axs[0].set_title('Y vs Z')
    axs[0].set_xlabel('Z')
    axs[0].set_ylabel('Y')
    axs[0].set_xlim(max(points_array[:,2])+30, -10)
    axs[0].set_ylim(max(points_array[:,1])+30, min(points_array[:,1])-30)
    axs[0].legend(["Ball Location", "Left Camera", "Catcher", "Right Camera","Best Fit Parabola"])
    axs[0].grid()

    axs[1].scatter(points_array[:,2], points_array[:,0])
    axs[1].plot(left_camera_location[2], left_camera_location[0], 'ro')
    axs[1].plot(catcher_location[2], catcher_location[0], 'ko')
    axs[1].plot(right_camera_location[2], right_camera_location[0], 'bo')
    # plot the xz model
    z = np.linspace(max(points_array[:,2])+30, 0, 100)
    x = xz_model[0]*z + xz_model[1]
    axs[1].plot(z, x, 'r')
    axs[1].set_xlim(max(points_array[:,2])+30, -10)
    axs[1].set_ylim(-40, 40)
    axs[1].set_title('X vs Z')
    axs[1].set_xlabel('Z')
    axs[1].set_ylabel('X')
    axs[1].legend(["Ball Location", "Left Camera", "Catcher", "Right Camera", "Best Fit Line"])
    axs[1].grid()

    plt.savefig("hw5_baseballtracking/output_imgs/3D_Ball_Location.png")
    plt.show()