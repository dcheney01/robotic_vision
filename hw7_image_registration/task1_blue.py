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

    circles_x = circles[0][np.argsort(circles[0][:, 0])]
    stds[1] = np.std(circles_x[:6, 0])
    stds[3] = np.std(circles_x[-6:, 0])

    circles_y = circles[0][np.argsort(circles[0][:, 1])]
    stds[0] = np.std(circles_y[:6, 1])
    stds[2] = np.std(circles_y[-6:, 1])

    return np.argmin(stds) * 90

def orient_img(img):
    img_circles = get_circle_anchors(img)
    rotation = get_orientation(img_circles)

    if rotation != 0:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE if rotation == 90 else cv.ROTATE_180 if rotation == 180 else cv.ROTATE_90_COUNTERCLOCKWISE if rotation == 270 else cv.ROTATE_180)

    return img

def get_answer_grid(cropped_abs_diff_processed, oriented_img_cropped, display=False):
    vertical_lines = [3, 27, 51, 75, 99, 123, 147, 171, 195, 219, 243, 267, 291, 315, 339, 363, 387, 411, 435, 459, 483]
    section1_horizontal = [19, 31, 43, 56, 67, 79, 91, 103, 115, 128, 141]
    section2_horizontal = [173, 186, 198, 210, 222, 234, 246, 258, 270, 282, 294]
    section3_horizontal = [326, 339, 351, 363, 375, 387, 399, 411, 423, 435, 447]
    extracted_answers = np.zeros((60, 10))

    for y in range(len(vertical_lines)-1, 0, -1):
        answer_idx_y = np.abs(20 - y)

        for x in range(len(section1_horizontal)-1):
            extracted_answers[answer_idx_y][x] = cropped_abs_diff_processed[section1_horizontal[x]:section1_horizontal[x+1], vertical_lines[y-1]:vertical_lines[y]].sum()
            if display:
                cv.rectangle(cropped_abs_diff_processed, (vertical_lines[y-1], section1_horizontal[x]), (vertical_lines[y], section1_horizontal[x+1]), (0, 255, 0), 1)
                cv.rectangle(oriented_img_cropped, (vertical_lines[y-1], section1_horizontal[x]), (vertical_lines[y], section1_horizontal[x+1]), (0, 255, 0), 1)

        for x in range(len(section2_horizontal)-1):
            extracted_answers[answer_idx_y+20][x] = cropped_abs_diff_processed[section2_horizontal[x]:section2_horizontal[x+1], vertical_lines[y-1]:vertical_lines[y]].sum()
            if display:
                cv.rectangle(cropped_abs_diff_processed, (vertical_lines[y-1], section2_horizontal[x]), (vertical_lines[y], section2_horizontal[x+1]), (0, 255, 0), 1)
                cv.rectangle(oriented_img_cropped, (vertical_lines[y-1], section2_horizontal[x]), (vertical_lines[y], section2_horizontal[x+1]), (0, 255, 0), 1)
        
        for x in range(len(section3_horizontal)-1):
            extracted_answers[answer_idx_y+40][x] = cropped_abs_diff_processed[section3_horizontal[x]:section3_horizontal[x+1], vertical_lines[y-1]:vertical_lines[y]].sum()
            if display:
                cv.rectangle(cropped_abs_diff_processed, (vertical_lines[y-1], section3_horizontal[x]), (vertical_lines[y], section3_horizontal[x+1]), (0, 255, 0), 1)
                cv.rectangle(oriented_img_cropped, (vertical_lines[y-1], section3_horizontal[x]), (vertical_lines[y], section3_horizontal[x+1]), (0, 255, 0), 1)

    return extracted_answers

def print_answers(answers):
    for i in range(60):
        print(f"Question {i+1}: {answers[i]}")
        if (i+1) % 20 == 0:
            print()

def save_answers(img_name, answers):
    with open(f'hw7_image_registration/output_data/Blue Output/Blue Output {img_name}.txt', 'w') as f:
        for i in range(60):
            f.write(f"{answers[i]}\n")

if __name__=="__main__":
    blue_imgs_path = os.path.join('hw7_image_registration', 'data', 'blue')

    blue_original = cv.imread(os.path.join(blue_imgs_path, 'Blue.jpg'))
    blue_original = cv.resize(blue_original, (blue_original.shape[1]//4, blue_original.shape[0]//4))
    blue_original_gray = cv.cvtColor(blue_original, cv.COLOR_BGR2GRAY)

    original_circles = get_circle_anchors(blue_original)
    ind = np.lexsort((original_circles[0][:, 1], original_circles[0][:, 0]))
    original_circles = original_circles[0][ind]
    blue_original_cropped = blue_original[int(blue_original.shape[0]/2.5):, 40:blue_original.shape[1]-110]

    KEY = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

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

            abs_diff = cv.absdiff(blue_original_gray, cv.cvtColor(oriented_img, cv.COLOR_BGR2GRAY))
            cv.imwrite('hw7_image_registration/output_data/Blue Output/Blue Output ' + img_path.split(" ")[-1], abs_diff)

            oriented_img_cropped = oriented_img[int(oriented_img.shape[0]/2.5):, 40:oriented_img.shape[1]-110]
            cropped_abs_diff = cv.absdiff(blue_original_cropped, oriented_img_cropped)
            _, cropped_abs_diff_processed = cv.threshold(cropped_abs_diff, 120, 255, cv.THRESH_BINARY)

            answer_grid = get_answer_grid(cropped_abs_diff_processed, oriented_img_cropped, display=True)
            
            answer_idxs = np.argmax(answer_grid, axis=1)
            answers = [KEY[i] for i in answer_idxs]
            print_answers(answers)
            save_answers(img_path.split(" ")[-1].split(".")[0], answers)

            cv.imshow('cropped_abs_diff_processed', cropped_abs_diff_processed)
            cv.imshow('oriented_img', oriented_img_cropped)
            cv.imwrite('hw7_image_registration/output_data/cropped_abs_diff_rectangles ' + img_path.split(" ")[-1], cropped_abs_diff_processed)
            cv.waitKey(0)
