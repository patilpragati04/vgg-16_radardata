import cv2
from PIL import ImageOps, Image
from keras.utils import img_to_array
from keras.utils import load_img
import os
import imutils

import pandas as pd

BASE_OUTPUT = "output"

BASE_PATH = 'dataset'
# train_File = 'train_Images'
Train_IMAGES_PATH = os.path.sep.join([BASE_PATH, "cropped_train_Images"])
cropped_Image_PATH = os.path.sep.join([BASE_PATH, "cropped_train_Images"])
Train_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "train_annotations.xml"])
train_imagePath ='dataset/cropped_validation_Images/11-08-2022 12-37-11-883.jpg'
predictedimage_PATH = os.path.sep.join([BASE_OUTPUT, "predictedimage.png"])

xmin =int(345)
ymin=int(420)
xmax=int(372)
ymax =int(443)
# for image in os.listdir(Train_IMAGES_PATH):
# # print(image)
#     train_imagePath = os.path.sep.join([Train_IMAGES_PATH, image])
    # cropped_img = load_img(train_imagePath)
    # # print(cropped_img.shape)
    # cropped_img =img_to_array(cropped_img)
# to resize all the images to same size
#     cropped_img =cv2.imread(train_imagePath)
#     cropped_img =cv2.resize(cropped_img,(840,630))


# cropped_img.show()
# cropped_img =  cv2.rectangle(cropped_img,(xmin,ymin),(xmax,ymax),(0,0,255),2)
#     cv2.imshow(cropped_img)
#     cropped_img =cv2.imread(cropped_img)

    # border = (109, 50, 80, 70) # left, top, right, bottom
    # cropped_img = ImageOps.crop(cropped_img, border)
    # # cropped_img.show()
    # cropped_img.save(cropped_Image_PATH +"/"+ image )
    # cv2.imwrite(cropped_Image_PATH +"/"+ image ,cropped_img)


# for image in root:
# new_image= int(image.width)
#     rank.text = str(new_rank)
#     rank.set('changed', 'yes')
#
# tree.write('change.xml')

# # print("HH")
cropped_img =load_img(train_imagePath)
cropped_img = img_to_array(cropped_img)
# print("JJJJ")
# # image = imutils.resize(image, width=224)
cv2.rectangle(cropped_img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
print("KKKK")
cv2.imshow("output", cropped_img)
cv2.imwrite(predictedimage_PATH,cropped_img)