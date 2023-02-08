# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
train_data = []
classes = ['Human','No Human']
trainFilenames = []
train_imagePaths = []
labels = []
trainbboxes = []
# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])
BASE_PATH = "dataset"
Train_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "JOB2/Annotations"])
image_extension = "jpg"
xmls = os.listdir(Train_ANNOTS_PATH)
IOU = open(r'info/IOU.txt', 'w')

def convert_annotation(image_id,list_file):
    in_file = open('dataset/JOB2/Annotations/%s.xml' % image_id)
    tree = ET.parse(in_file)
    root = tree.getroot()
    object_detected = False
    for child in root:
        # print(child.tag, child.attrib)
        Filenames = root.find('filename').text
        if child.tag == 'object':
            cls = child.find('name').text
            labels.append(cls)
            difficult = child.find('difficult').text
            object_detected = True
            cls_id = classes.index(cls)
            xmlbox = child.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
            float(xmlbox.find('ymax').text))
            trainbboxes.append(b)
            list_file.write(" " + ",".join([str(a) for a in b]) )#+ ','+str(cls_id)

    if object_detected == False:
        cls = "No Human"
        labels.append(cls)
        cls_id = classes.index(cls)
        difficult = 1
        b = (0, 0, 0, 0)
        trainbboxes.append(b)
        list_file.write(" " + ",".join([str(a) for a in b]) ) #+ ','+str(cls_id)
    # list_file.write('\n')
    return list_file

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou
for xml in xmls:
    # print(xml)
    name = xml[:-4]
    a = name.replace(" ","_")
    IOU.write('dataset/JOB2/JPEGImages/%s.%s' % (name, image_extension))
    IOU = convert_annotation(name,IOU)
    IOU.write('\n')
# for images in ImagePath:
# define the list of example detections
examples = [
	Detection("dataset/Predidction_data/JPEGImages/11-08-2022_12-36-15-705.jpg", [449.657, 66, 534, 106], [int(0.3504865),  int(0.3111757) , int(0.38172194) ,int(0.40643385)]),
	# Detection("image2.jpg", [49, 75, 203, 125], [449.27, 66.6, 534.43, 106.67]),
	# Detection("image10.jpg", [31, 69, 201, 125], [18, 63, 235, 135]),
	# Detection("image5.jpg", [50, 72, 197, 121], [54, 72, 198, 120]),
	# Detection("image4.jpg", [35, 51, 196, 110], [36, 60, 180, 108])
]
# loop over the example detections
for detection in examples:
	# load the image
	image = cv2.imread(detection.image_path)
	# draw the ground-truth bounding box along with the predicted
	# bounding box
	cv2.rectangle(image, tuple(detection.gt[:2]),
		tuple(detection.gt[2:]), (0, 255, 0), 2)
	print(detection.gt[:2])
	cv2.rectangle(image, tuple(detection.pred[:2]),
		tuple(detection.pred[2:]), (0, 0, 255), 2)
	print(detection.pred[:2])

	# compute the intersection over union and display it
	iou = bb_intersection_over_union(detection.gt, detection.pred)
	cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	print("{}: {:.4f}".format(detection.image_path, iou))
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)

