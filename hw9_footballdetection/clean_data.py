# import cv2

# video_path = "data/Train 1.mp4"
# imgs_path = "data/images/"

# cap = cv2.VideoCapture(video_path)
# # cap.set(cv2.CAP_PROP_POS_FRAMES, 1000 * i)

# # save images from video to folder
# for i in range(0, 551):
#     ret, frame = cap.read()
#     # write names as frame_000000.jpg
#     cv2.imwrite(imgs_path + "frame_{:06d}.jpg".format(i), frame)
# cap.release()

import splitfolders

input_path = "data/raw/"

splitfolders.ratio(input_path, output="data", seed=1337, ratio=(.8, .2)) # default values
