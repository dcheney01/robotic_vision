import os
import numpy as np
import cv2
import time

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
    return x-15, y-15, 50, 50

def find_ball(x, y, w, h, img, roi_image):
    max_y = y+h if y+h < img.shape[0] else img.shape[0]
    max_x = x+w if x+w < img.shape[1] else img.shape[1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_roi_gray = gray[y:max_y, x:max_x]
    circles = cv2.HoughCircles(img_roi_gray, cv2.HOUGH_GRADIENT, 4.89, 10000, param1=250, param2=15, minRadius=1, maxRadius=13)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(roi_image, (i[0]+x, i[1]+y), i[2], (0, 255, 0), 2)
            cv2.circle(roi_image, (i[0]+x, i[1]+y), 2, (0, 0, 255), 3)
    else:
        print("No circles found")
    
    # cv2.imshow("Circles", roi_image)
    return roi_image

def run_task2(data_path, LR="L"):
    img_folder = data_path + LR + "/"
    img_paths = sorted(os.listdir(img_folder), key=lambda x: int(x.split(".")[0]))

    prev_image = cv2.imread(img_folder + img_paths[0])

    counter = 1
    for img_path in img_paths:
        print(f"Processing: {img_path}")
        start = time.time()
        full_img_path = img_folder + img_path
        img = cv2.imread(full_img_path)

        contours = find_contours(img, prev_image)

        if len(contours) > 0:
            x, y, w, h = get_roi_box(contours)
            roi_image = img.copy()
            cv2.rectangle(roi_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow('Original Image with ROI', roi_image)

            ball_img = find_ball(x, y, w, h, img, roi_image)
        else:
            print("Didn't find any contours")
        
        prev_image = img

        print(f"Loop Time: {time.time() - start}")
        # cv2.imshow("Left Image", img)
        # cv2.waitKey(0)

        if counter % 5 == 0:
            cv2.imwrite("robotic_vision/hw5_baseballtracking/output_imgs/" + LR + str(counter) + ".png", ball_img)
        counter += 1

if __name__=="__main__":
    data_path = "robotic_vision/hw5_baseballtracking/data/20240215113225/"
    
    run_task2(data_path, "L")

    run_task2(data_path, "R")


