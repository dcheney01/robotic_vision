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

def get_object_location(good, ref_kp, target_kp, h, w):
    src_points = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([target_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # find the homography matrix
    M, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    return matchesMask, dst

if __name__=="__main__":
    ref_img_path = os.path.join('hw6_homography', 'data', 'ref_img.jpg')

    ref_img = cv.imread(ref_img_path)
    ref_img_cropped = ref_img[:475, 125:590]
    ref_img_cropped_gray = cv.cvtColor(ref_img_cropped, cv.COLOR_BGR2GRAY)
    h, w = ref_img_cropped.shape[:2]

    # cv.imshow('ref_img_cropped', ref_img_cropped)
    # cv.imshow('img_with_ref', img_with_ref)
    # cv.waitKey(0)

    # setup sift and flann + drawing params (constant)
    sift = cv.SIFT_create()
    index_params = dict(algorithm=1)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    ref_kp, ref_des = get_keypoints(ref_img_cropped_gray, sift)

    # for all images
    img_with_ref = os.path.join('hw6_homography', 'data', 'ref_img_in_scene.jpg')
    img_with_ref = cv.imread(img_with_ref)
    img_with_ref_gray = cv.cvtColor(img_with_ref, cv.COLOR_BGR2GRAY)
    target_kp, good = get_matches(img_with_ref_gray, ref_des, sift, flann)
    matchesMask, dst = get_object_location(good, ref_kp, target_kp, h, w)

    # draw object outline and the matches
    img_with_ref = cv.polylines(img_with_ref, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
    img_matches = cv.drawMatches(ref_img_cropped, ref_kp, img_with_ref, target_kp, good, None, **draw_params)

    cv.imshow('img_matches', img_matches)
    cv.imwrite('hw6_homography/output/task1_matches.jpg', img_matches)
    cv.waitKey(0)

