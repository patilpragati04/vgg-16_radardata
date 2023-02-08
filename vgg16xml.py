# example of extracting bounding boxes from an annotation file
from xml.etree import ElementTree
# import the necessary packages
from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import img_to_array
from keras.utils import load_img
import keras
import elementpath
from keras.metrics import Accuracy, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC, BinaryAccuracy
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims
import cv2
import os
import pickle
from os import listdir
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings('ignore')
BASE_PATH = "dataset"
Train_IMAGES_PATH = os.path.sep.join([BASE_PATH, "JOB1/JPEGImages/"])
Train_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "JOB1/Annotations/"])
Validation_IMAGES_PATH = os.path.sep.join([BASE_PATH, "JOB2/JPEGImages/"])
Validation_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "JOB2/Annotations/"])
# and testing image filenames
BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
LOSS_PATH = os.path.sep.join([BASE_OUTPUT, "loss.png"])
ACCURACY_PATH = os.path.sep.join([BASE_OUTPUT, "acc.png"])
FEATUREMAP_PATH = os.path.sep.join([BASE_OUTPUT, "featuremap.png"])
# TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])
INIT_LR = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 32
train_data = []
train_images = []
validation_images =[]
validation_data = []

# function to extract bounding boxes from an annotation file
for filename in listdir(Train_IMAGES_PATH):
	# extract image id
	image_id = filename[:-4]
	img_path = 'dataset/JOB1/JPEGImages/10-27-2022_14-29-16-834.jpg'
	ann_path = 'dataset/JOB1/Annotations/10-27-2022_14-29-16-834.xml'
	train_image =cv2.imread(img_path)

	(h, w) = train_image.shape[:2]
	# ann_path = os.path.join(BASE_PATH,'JOB1/Annotations/')
	train_image = load_img(img_path, target_size=(224, 224))
	train_image = img_to_array(train_image)
	print(len(train_image),"jjjj")

	# train_images.append(train_image)
	# train_images = np.array(train_images, dtype="float32") / 255.0

	# load and parse the file
	tree = ElementTree.parse(ann_path)
	# get the root of the document
	root = tree.getroot()
	# extract each bounding box
	boxes = list()
	if root.findall('.//bndbox') == None:
		xmin = 0
		ymin = 0
		xmax = 0
		ymax = 0
	else:
		for box in root.findall('.//bndbox'):
			xmin = float(box.find('xmin').text)
			ymin = float(box.find('ymin').text)
			xmax = float(box.find('xmax').text)
			ymax = float(box.find('ymax').text)
			coors = [xmin /w, ymin/h, xmax/ w, ymax/h]
			boxes.append(coors)
		# extract image dimensions
		train_data.append(boxes)
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
	# train_images = np.array(train_images, dtype="float32") / 255.0
print((len(train_image)),"kkk")
# train_data = np.array(train_data, dtype="float32")

# for filename in listdir(Validation_IMAGES_PATH):
# 	# extract image id
# 	image_id = filename[:-4]
# 	img_path = Validation_IMAGES_PATH + filename
# 	ann_path = Validation_ANNOTS_PATH + image_id + '.xml'
# 	validation_image = load_img(img_path, target_size=(224, 224))
# 	validation_image = img_to_array(validation_image)
# 	validation_images.append(validation_image)
# 	# load and parse the file
# 	tree = ElementTree.parse(ann_path)
# 	# get the root of the document
# 	root = tree.getroot()
# 	# extract each bounding box
# 	boxes = list()
# 	for box in root.findall('.//bndbox'):
# 		xmin = float(box.find('xmin').text)
# 		ymin = float(box.find('ymin').text)
# 		xmax = float(box.find('xmax').text)
# 		ymax = float(box.find('ymax').text)
# 		coors = [xmin, ymin, xmax, ymax]
# 		boxes.append(coors)
# 	# extract image dimensions
# 	validation_data.append(boxes)
# 	width = int(root.find('.//size/width').text)
# 	height = int(root.find('.//size/height').text)
# validation_images = np.array(validation_images, dtype="float32") / 255.0
# # if xtl != "":
# 	bboxes = np.array(bboxes, dtype="float32")
# validation_data = np.array(validation_data, dtype="float32")
# print(len(validation_images))
# print(len(validation_data))
# print("Going inside the model")
# vgg = VGG16(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(224, 224, 3)))
# # freeze all VGG layers so they will *not* be updated during the
# # training process
# vgg.trainable = False
# # flatten the max-pooling output of VGG
# flatten = vgg.output
# flatten = Flatten()(flatten)
# # construct a fully-connected layer header to output the predicted
# # bounding box coordinates
# bboxHead = Dense(128, activation="relu")(flatten)
# bboxHead = Dense(64, activation="relu")(bboxHead)
# bboxHead = Dense(32, activation="relu")(bboxHead)
# bboxHead = Dense(4, activation="sigmoid",
# 	name="bounding_box")(bboxHead)
# # construct a second fully-connected layer head, this one to predict
# # the class label
# softmaxHead = Dense(512, activation="relu")(flatten)
# softmaxHead = Dropout(0.5)(softmaxHead)
# softmaxHead = Dense(512, activation="relu")(softmaxHead)
# softmaxHead = Dropout(0.5)(softmaxHead)
# softmaxHead = Dense(2, activation="sigmoid",
# 	name="class_label")(softmaxHead)
# # put together our model which accept an input image and then output
# # bounding box coordinates and a class label
# model = Model(
# 	inputs=vgg.input,
# 	outputs=(bboxHead))
# # This callback will stop the training when there is no improvement in
# # the loss for three consecutive epochs.
# early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=3,mode='auto')
# callback = keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# # initialize the optimizer, compile the model, and show the model
# # summary
# opt = Adam(lr=INIT_LR)
# metrics = [TruePositives(name='tp'),TrueNegatives(name='tn'), FalseNegatives(name='fn'), FalsePositives(name='fp'),BinaryAccuracy(name='accuracy'),Recall(name='recall'),Precision(name='precision'),AUC(name='auc')]
# model.compile(loss="mse", optimizer=opt, metrics=metrics)
# print(model.summary())
# # prediction
# print("[INFO] training model...")
# print("[INFO] training bounding box regressor...")
# # H = model.fit(
# # 	train_images, train_data,
# # 	validation_data=(validation_images, validation_data),
# # 	steps_per_epoch=BATCH_SIZE,
# # 	epochs=NUM_EPOCHS,callbacks=[callback,early_stopping],
# # 	verbose=1)
#
# stopped_epoch = (early_stopping.stopped_epoch+1)
# print(stopped_epoch)
# # serialize the model to disk
# print("[INFO] saving object detector model...")
#
#
# # serialize the label binarizer to disk
# print("[INFO] saving label binarizer...")
# f = open(LB_PATH, "wb")
# # f.write(pickle.dumps(lb))
# f.close()
# # plot the total loss, label loss, and bounding box loss
# lossNames = ["loss"]
# #N = np.arange(0, stopped_epoch)
# plt.style.use("ggplot")
# (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
#
# # create a new figure for the accuracies
# N = stopped_epoch
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, stopped_epoch), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, stopped_epoch), H.history["val_loss"], label="val_loss")
# plt.title("Bounding Box Regression Loss on Training Set")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend(loc="lower left")
# plt.savefig(LOSS_PATH)
# print("[INFO] saved loss Figure")
# plt.close()
# # #create a new figure for the accuracies
# N = stopped_epoch
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, stopped_epoch), H.history["accuracy"], label="Accuracy")
# plt.plot(np.arange(0, stopped_epoch), H.history["val_accuracy"], label="val_Accuracy")
# plt.title("Bounding Box Regression Loss on Training Set")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(ACCURACY_PATH)
# print("[INFO] saved accuracy Figure")
# plt.close()
#





