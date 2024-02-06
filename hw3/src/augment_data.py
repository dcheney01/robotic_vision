import albumentations as A
import cv2 as cv
import os

data_path = '/home/daniel/software/robotic_vision/hw3/data/Oyster Shell/train'
# data_path = '/home/daniel/software/robotic_vision/hw3/data/8 Fish Species/train'

transform = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.4),
    A.InvertImg(p=0.5),
    A.GaussNoise(p=0.3),
    A.GaussianBlur(p=0.3),
])

imgs_count = 0
for img_folder in os.listdir(data_path):
    for img in os.listdir(os.path.join(data_path, img_folder)):
        if img.split('.')[-1] == 'png' or img.split('.')[-1] == 'tif':
            original_image = cv.imread(os.path.join(data_path, img_folder, img))
            original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
            for i in range(4):
                transformed = transform(image=original_image)
                transformed_image = transformed['image']
                new_img_name = img.split('.')[0] + '_aug' + str(i) + '.jpg'
                cv.imwrite(os.path.join(data_path, img_folder, new_img_name), transformed_image)
                imgs_count += 1
        
print(f"Total images created: {imgs_count}")
        # if img.split('.')[-1] == "jpg":
        #     os.remove(os.path.join(data_path, img_folder, img))