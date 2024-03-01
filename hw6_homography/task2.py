import cv2 as cv
import numpy as np
import os

def get_keypoints(img, sift):
    kp, des = sift.detectAndCompute(img, None)
    des = np.float32(des)
    return kp, des

def get_matches(target, ref_des, sift, flann):
    target_kp, target_des = get_keypoints(target, sift)

    matches = flann.knnMatch(ref_des, target_des, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good.append(m)

    return target_kp, good

def get_homography_mask(good, ref_kp, target_kp):
    src_points = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([target_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # find the homography matrix
    M, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    return M, matchesMask

def transform_img(img, M):
    points = np.float32([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]]).reshape(-1, 1, 2)
    dst_points = cv.perspectiveTransform(points, M)
    dst_img = cv.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return dst_points, dst_img

if __name__=="__main__":
    ref_img_path = os.path.join('hw6_homography', 'data', 'ref_window.jpg')
    ref_img = cv.imread(ref_img_path)
    ref_img_cropped = ref_img[30:430, 85:610]
    ref_img_cropped_gray = cv.cvtColor(ref_img_cropped, cv.COLOR_BGR2GRAY)
    h, w = ref_img_cropped.shape[:2]

    target_img_path = os.path.join('hw6_homography', 'data', 'target_img_shire.jpg')
    target_img = cv.imread(target_img_path)
    target_img = cv.resize(target_img, (640, 480))

    video_path = os.path.join('hw6_homography', 'data', 'orig_video.mp4')
    # resize the video to 640, 480 as mp4
    cap = cv.VideoCapture(video_path)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('hw6_homography/output/task2_final_video1.mp4', fourcc, 30.0, (640, 480))

    # setup sift and flann
    sift = cv.SIFT_create()
    index_params = dict(algorithm=1)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    ref_kp, ref_des = get_keypoints(ref_img_cropped_gray, sift)

    for i in range(0, 50000):
        ret, scene = cap.read()
        if not ret:
            break

        scene_resized = cv.resize(scene, (640, 480))
        scene_gray = cv.cvtColor(scene_resized, cv.COLOR_BGR2GRAY)
        target_kp, good = get_matches(scene_gray, ref_des, sift, flann)
        M, matchesMask = get_homography_mask(good, ref_kp, target_kp)

        dst_points, dst_img = transform_img(target_img, M)

        # superimpose target_dst onto scene_resized
        target_dst = scene_resized.copy()
        cv.fillPoly(target_dst, [np.int32(dst_points)], (0, 0, 0))
        target_dst = cv.add(target_dst, dst_img)

        # draw object outline and the matches
        # scene_resized = cv.polylines(scene_resized, [np.int32(dst_points)], True, 255, 3, cv.LINE_AA)
        # draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
        # img_matches = cv.drawMatches(ref_img_cropped, ref_kp, scene_resized, target_kp, good, None, **draw_params)

        out.write(target_dst)

        # cv.imshow('img_matches', img_matches)
        # cv.imshow('target_dst', target_dst)
        # cv.waitKey(0)




