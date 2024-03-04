import numpy as np
import cv2 as cv
import os

def get_circle_anchors(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT_ALT, 1.25, 55, param1=100, param2=0.68, minRadius=8, maxRadius=14)
    circles = np.uint16(np.around(circles))
    return circles

def get_orientation(circles):
    # there are 6 circles in a row. They can have similar xs, or similar ys
    # if the lower xs are similar, its a 90 deg rotation
    # if the lower ys are similar, its a 0 deg rotation
    # if the upper xs are similar, its a 270 deg rotation
    # if the upper ys are similar, its a 180 deg rotation
    stds = np.zeros((4, 1))

    # sort by x position
    circles_x = circles[0][np.argsort(circles[0][:, 0])]
    # print(circles_x)
    # print(circles_x[6:, 0])
    stds[1] = np.std(circles_x[:6, 0])
    stds[3] = np.std(circles_x[-6:, 0])

    circles_y = circles[0][np.argsort(circles[0][:, 1])]
    stds[0] = np.std(circles_y[:6, 1])
    stds[2] = np.std(circles_y[-6:, 1])

    # print(stds)

    return np.argmin(stds) * 90

def orient_img(img):
    # get the circle anchors to figure out the relative orientation
    img_circles = get_circle_anchors(img)
    rotation = get_orientation(img_circles)

    if rotation != 0:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE if rotation == 90 else cv.ROTATE_180 if rotation == 180 else cv.ROTATE_90_COUNTERCLOCKWISE if rotation == 270 else cv.ROTATE_180)

    return img

if __name__=="__main__":
    blue_imgs_path = os.path.join('hw7_image_registration', 'data', 'blue')

    blue_original = cv.imread(os.path.join(blue_imgs_path, 'Blue.jpg'))
    blue_original = cv.resize(blue_original, (blue_original.shape[1]//4, blue_original.shape[0]//4))
    blue_original_gray = cv.cvtColor(blue_original, cv.COLOR_BGR2GRAY)

    original_circles = get_circle_anchors(blue_original)
    ind = np.lexsort((original_circles[0][:, 1], original_circles[0][:, 0]))
    original_circles = original_circles[0][ind]

    # crop the top third off of the image
    # blue_original_cropped = blue_original[int(blue_original.shape[0]/2.5):, :, :]
    # cv.imshow('blue_original', blue_original)
    # cv.waitKey(0)

    # cv.imshow('blue_original', blue_original)

    for img_path in os.listdir(blue_imgs_path):
        if img_path == 'Blue.jpg':
            continue
        else:
            print(img_path)
            img_path_full = os.path.join(blue_imgs_path, img_path)
            img = cv.imread(img_path_full)
            img = cv.resize(img, (img.shape[1]//4, img.shape[0]//4))
            
            oriented_img = orient_img(img)
            oriented_img = cv.resize(oriented_img, (blue_original.shape[1], blue_original.shape[0]))
            # oriented_circles = get_circle_anchors(oriented_img)
            # ind = np.lexsort((oriented_circles[0][:, 1], oriented_circles[0][:, 0]))
            # oriented_circles = oriented_circles[0][ind]

            abs_diff = cv.absdiff(blue_original_gray, cv.cvtColor(oriented_img, cv.COLOR_BGR2GRAY))
            cv.imwrite('hw7_image_registration/output_data/Blue Output ' + img_path.split(" ")[-1], abs_diff)

            oriented_img_cropped = oriented_img[int(oriented_img.shape[0]/2.5):, 40:oriented_img.shape[1]-110]

            # cropped_circles = get_circle_anchors(oriented_img_cropped)

            cv.imshow('oriented_img', oriented_img_cropped)
            cv.waitKey(0)
            
            
        
            
            
            # cv.imshow('abs_diff', abs_diff)
            # cv.imshow('oriented_img', oriented_img)
            # cv.waitKey(0)
