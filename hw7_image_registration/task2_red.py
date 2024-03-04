import numpy as np
import cv2 as cv
import os


def plot_lines(cropped_abs_diff_processed, img_cropped, horizontal_lines, section1_vertical, section2_vertical, section3_vertical, section4_vertical):
    for y in horizontal_lines:
        cv.line(cropped_abs_diff_processed, (0, y), (cropped_abs_diff_processed.shape[1], y), (0, 255, 0), 1)
        cv.line(img_cropped, (0, y), (img_cropped.shape[1], y), (0, 255, 0), 1)

    for x in section1_vertical:
        cv.line(cropped_abs_diff_processed, (x, 0), (x, cropped_abs_diff_processed.shape[0]), (0, 255, 0), 1)
        cv.line(img_cropped, (x, 0), (x, img_cropped.shape[0]), (0, 255, 0), 1)

    for x in section2_vertical:
        cv.line(cropped_abs_diff_processed, (x, 0), (x, cropped_abs_diff_processed.shape[0]), (255, 255, 0), 1)
        cv.line(img_cropped, (x, 0), (x, img_cropped.shape[0]), (255, 255, 0), 1)

    for x in section3_vertical:
        cv.line(cropped_abs_diff_processed, (x, 0), (x, cropped_abs_diff_processed.shape[0]), (0, 255, 255), 1)
        cv.line(img_cropped, (x, 0), (x, img_cropped.shape[0]), (0, 255, 255), 1)

    for x in section4_vertical:
        cv.line(cropped_abs_diff_processed, (x, 0), (x, cropped_abs_diff_processed.shape[0]), (255, 0, 255), 1)
        cv.line(img_cropped, (x, 0), (x, img_cropped.shape[0]), (255, 0, 255), 1)

def get_answer_grid(cropped_abs_diff_processed, img_cropped, display=False):
    horizontal_lines = np.linspace(6, cropped_abs_diff_processed.shape[0], num=26, dtype=int)
    one_start = 11
    two_start = 160
    three_start = 310
    four_start = 460
    space = 125
    section1_vertical = np.linspace(one_start, one_start+space, num=11, dtype=int)
    section2_vertical = np.linspace(two_start, two_start+space, num=11, dtype=int)
    section3_vertical = np.linspace(three_start, three_start+space, num=11, dtype=int)
    section4_vertical = np.linspace(four_start, four_start+space, num=11, dtype=int)

    extracted_answers = np.zeros((100, 10))

    for y in range(len(horizontal_lines)-1):
        for x in range(len(section1_vertical)-1):
            extracted_answers[y][x] = cropped_abs_diff_processed[horizontal_lines[y]:horizontal_lines[y+1], section1_vertical[x]:section1_vertical[x+1]].sum()
            if display:
                cv.rectangle(cropped_abs_diff_processed, (section1_vertical[x], horizontal_lines[y]), (section1_vertical[x+1], horizontal_lines[y+1]), (0, 255, 0), 1)
                cv.rectangle(img_cropped, (section1_vertical[x], horizontal_lines[y]), (section1_vertical[x+1], horizontal_lines[y+1]), (0, 255, 0), 1)

        for x in range(len(section2_vertical)-1):
            extracted_answers[y+25][x] = cropped_abs_diff_processed[horizontal_lines[y]:horizontal_lines[y+1], section2_vertical[x]:section2_vertical[x+1]].sum()
            if display:
                cv.rectangle(cropped_abs_diff_processed, (section2_vertical[x], horizontal_lines[y]), (section2_vertical[x+1], horizontal_lines[y+1]), (255, 255, 0), 1)
                cv.rectangle(img_cropped, (section2_vertical[x], horizontal_lines[y]), (section2_vertical[x+1], horizontal_lines[y+1]), (255, 255, 0), 1)

        for x in range(len(section3_vertical)-1):
            extracted_answers[y+50][x] = cropped_abs_diff_processed[horizontal_lines[y]:horizontal_lines[y+1], section3_vertical[x]:section3_vertical[x+1]].sum()
            if display:
                cv.rectangle(cropped_abs_diff_processed, (section3_vertical[x], horizontal_lines[y]), (section3_vertical[x+1], horizontal_lines[y+1]), (0, 255, 255), 1)
                cv.rectangle(img_cropped, (section3_vertical[x], horizontal_lines[y]), (section3_vertical[x+1], horizontal_lines[y+1]), (0, 255, 255), 1)

        for x in range(len(section4_vertical)-1):
            extracted_answers[y+75][x] = cropped_abs_diff_processed[horizontal_lines[y]:horizontal_lines[y+1], section4_vertical[x]:section4_vertical[x+1]].sum()
            if display:
                cv.rectangle(cropped_abs_diff_processed, (section4_vertical[x], horizontal_lines[y]), (section4_vertical[x+1], horizontal_lines[y+1]), (255, 0, 255), 1)
                cv.rectangle(img_cropped, (section4_vertical[x], horizontal_lines[y]), (section4_vertical[x+1], horizontal_lines[y+1]), (255, 0, 255), 1)
        
    return extracted_answers
        
def print_answers(answers):
    for i in range(100):
        print(f"Question {i+1}: {answers[i]}")
        if (i+1) % 25 == 0:
            print()

def save_answers(img_name, answers):
    with open(f'hw7_image_registration/output_data/Red Output/Red Output {img_name}.txt', 'w') as f:
        for i in answers:
            f.write(f"{i}\n")

if __name__=="__main__":
    red_imgs_path = os.path.join('hw7_image_registration', 'data', 'red')

    red_original = cv.imread(os.path.join(red_imgs_path, 'Red.jpg'))
    red_original = cv.resize(red_original, (red_original.shape[1]//4, red_original.shape[0]//4))
    red_original_cropped = red_original[170:red_original.shape[0]-25, 25:red_original.shape[1]-25]
   
    KEY = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    for img_path in os.listdir(red_imgs_path):
        if img_path == 'Red.jpg':
            continue
        else:
            print(img_path)
            img_path_full = os.path.join(red_imgs_path, img_path)
            img = cv.imread(img_path_full)
            img = cv.resize(img, (red_original.shape[1], red_original.shape[0]))
            img_cropped = img[170:img.shape[0]-25, 25:img.shape[1]-25]

            abs_diff = cv.absdiff(red_original, img)
            abs_diff_cropped = cv.absdiff(red_original_cropped, img_cropped)
            abs_diff_cropped[:,:,0] = 0
            abs_diff_cropped[:,:,1] = 0
            abs_diff[:,:,0] = 0
            abs_diff[:,:,1] = 0

            abs_diff_cropped_gray = cv.cvtColor(abs_diff_cropped, cv.COLOR_BGR2GRAY)
            abs_diff = cv.cvtColor(abs_diff, cv.COLOR_BGR2GRAY)
            _, cropped_abs_diff_processed = cv.threshold(abs_diff_cropped_gray, 30, 255, cv.THRESH_BINARY)
            _, abs_diff_processed = cv.threshold(abs_diff, 30, 255, cv.THRESH_BINARY)
            cropped_abs_diff_processed = cv.cvtColor(cropped_abs_diff_processed, cv.COLOR_GRAY2BGR)
            cv.imwrite('hw7_image_registration/output_data/Red Output/Red Output ' + img_path.split(" ")[-1], abs_diff_processed)

            answer_grid = get_answer_grid(cropped_abs_diff_processed, img_cropped, display=True)
            
            answer_idxs = np.argmax(answer_grid, axis=1)
            answers = [KEY[i] for i in answer_idxs]
            # print_answers(answers)
            save_answers(img_path.split(" ")[-1].split(".")[0], answers)

            # cv.imshow("abs_diff_cropped_gray", abs_diff_cropped_gray)
            # cv.imshow("abs_diff_cropped", abs_diff_cropped)
            # cv.imshow('red_original_cropped', red_original_cropped)
            # cv.imshow('img_cropped', img_cropped)
            # cv.imshow('cropped_abs_diff_processed', cropped_abs_diff_processed)
            # # cv.imwrite('hw7_image_registration/output_data/cropped_abs_diff_rectangles ' + img_path.split(" ")[-1], cropped_abs_diff_processed)
            # cv.waitKey(0)
            # # break