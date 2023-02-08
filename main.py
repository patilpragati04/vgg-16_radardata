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
from PIL import ImageOps, Image

# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file
BASE_PATH = 'dataset'
# train_File = 'train_Images'
Train_IMAGES_PATH = os.path.sep.join([BASE_PATH, "cropped_train_Images"])
# print(Train_IMAGES_PATH)
Train_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "cropped_train_annotations.csv"])
Validation_IMAGES_PATH = os.path.sep.join([BASE_PATH, "cropped_validation_Images"])
Validation_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "cropped_validation_annotations.csv"])
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
LOSS_PATH = os.path.sep.join([BASE_OUTPUT, "loss.png"])
ACCURACY_PATH = os.path.sep.join([BASE_OUTPUT, "acc.png"])
FEATUREMAP_PATH = os.path.sep.join([BASE_OUTPUT, "featuremap.png"])
# TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 32
# load the contents of the CSV annotations file
print("[INFO] loading dataset...")
train_rows = open(Train_ANNOTS_PATH).read().strip().split("\n")
validation_rows = open(Validation_ANNOTS_PATH).read().strip().split("\n")
# initialize the list of data (images), our target output predictions
# (bounding box coordinates), along with the filenames of the
# individual images
train_data = []
trainFilenames = []
train_imagePaths = []
validation_data = []
validationFilenames = []
validation_imagePaths = []
train_labels = []
trainbboxes = []
validationbboxes = []
validation_labels =[]
labels =[]
# loop over all CSV files in the annotations directory
# loop over the row

for train_row in train_rows:
	# break the row into the filename, bounding box coordinates,
	# and class label
	train_row = train_row.split(",")
	(name, width, height, label, xtl, ytl, xbr, ybr, z_order) = train_row
	# derive the path to the input image, load the image (in OpenCV
	# format), and grab its dimensions
	train_imagePath = os.path.sep.join([Train_IMAGES_PATH, name])
	#print(imagePath)
	image = cv2.imread(train_imagePath)

	# plt.imshow(image)
	# plt.show()
	(h, w) = image.shape[:2]
	# print(w,h)
	# scale the bounding box coordinates relative to the spatial
	# dimensions of the input image
	if xtl != "":
		xtl = float(xtl) / w
		ytl = float(ytl) / h
		xbr = float(xbr) / w
		ybr = float(ybr) / h
	else:
		xtl = 0
		ytl = 0
		xbr = 0
		ybr = 0
		label = "No Human"
	# load the image and preprocess it
	train_image = load_img(train_imagePath, target_size=(224, 224,3))
	train_image = img_to_array(train_image)
	# update our list of data, targets, and filenames
	train_data.append(train_image)
	labels.append(label)
	trainbboxes.append((xtl, ytl, xbr, ybr))
	trainFilenames.append(train_image)
	train_imagePaths.append(train_imagePath)
# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
# if xtl != "":
# 	bboxes = np.array(bboxes, dtype="float32")
train_data = np.array(train_data, dtype="float32") / 255.0
train_labels = np.array(labels)
train_boxes = np.array(trainbboxes, dtype="float32")
print(len(train_boxes))
print(len(train_data))
train_imagePaths = np.array(train_imagePath)
# print(data)
for validation_row in validation_rows:
	# break the row into the filename, bounding box coordinates,
	# and class label
	validation_row = validation_row.split(",")
	(name, width, height, label, xtl, ytl, xbr, ybr, z_order) = validation_row
	# derive the path to the input image, load the image (in OpenCV
	# format), and grab its dimensions
	validation_imagePath = os.path.sep.join([Validation_IMAGES_PATH, name])
	# print(validation_imagePath)
	# print(imagePath)
	image = cv2.imread(validation_imagePath)
	# plt.imshow(image)
	# plt.show()
	(h, w) = image.shape[:2]
	# scale the bounding box coordinates relative to the spatial
	# dimensions of the input image
	if xtl != "":
		xtl = float(xtl) / w
		ytl = float(ytl) / h
		xbr = float(xbr) / w
		ybr = float(ybr) / h
	else:
		xtl = 0
		ytl = 0
		xbr = 0
		ybr = 0
		label = "No Human"
	# load the image and preprocess it
	validation_image = load_img(validation_imagePath, target_size=(224, 224))
	validation_image = img_to_array(validation_image)
	# update our list of data, targets, and filenames
	validation_data.append(validation_image)
	labels.append(label)
	validationbboxes.append((xtl, ytl, xbr, ybr))
	validationFilenames.append(validation_image)
	validation_imagePaths.append(validation_imagePath)
# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
validation_data = np.array(validation_data, dtype="float32") / 255.0
# if xtl != "":
# 	bboxes = np.array(bboxes, dtype="float32")
validation_labels = np.array(labels)
validation_boxes = np.array(validationbboxes, dtype="float32")
train_imagePaths = np.array(validation_imagePath)
print(len(validation_boxes))
print(len(validation_data))

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# only there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well
if len(lb.classes_) == 2:
	labels = to_categorical(labels)
# load the VGG16 network, ensuring the head FC layers are left off
print("Going inside the model")
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid",
	name="bounding_box")(bboxHead)
# construct a second fully-connected layer head, this one to predict
# the class label
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(2, activation="sigmoid",
	name="class_label")(softmaxHead)
# put together our model which accept an input image and then output
# bounding box coordinates and a class label
model = Model(
	inputs=vgg.input,
	outputs=(bboxHead))
# This callback will stop the training when there is no improvement in
# the loss for three consecutive epochs.
early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=3,mode='auto')
callback = keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=INIT_LR)
metrics = [TruePositives(name='tp'),TrueNegatives(name='tn'), FalseNegatives(name='fn'), FalsePositives(name='fp'),BinaryAccuracy(name='accuracy'),Recall(name='recall'),Precision(name='precision'),AUC(name='auc')]
model.compile(loss="mse", optimizer=opt, metrics=metrics)
print(model.summary())
# prediction
print("[INFO] training model...")
print("[INFO] training bounding box regressor...")
print(len(train_data))
H = model.fit(
	train_data, train_boxes,
	validation_data=(validation_data, validation_boxes),
	steps_per_epoch=BATCH_SIZE,
	epochs=NUM_EPOCHS,callbacks=[callback,early_stopping],
	verbose=1)

stopped_epoch = (early_stopping.stopped_epoch+1)
print(stopped_epoch)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(MODEL_PATH, save_format="h5")
for i in range(len(model.layers)):
	layer = model.layers[i]
	if 'conv' not in layer.name:
		continue
	print(i, layer.name, layer.output.shape)
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
# expand dimensions so that it represents a single 'sample'
image = expand_dims(train_image, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
image = preprocess_input(image)
# get feature map for first hidden layer
feature_maps = model.predict(image)
fig = plt.figure(figsize=(20, 15))
for i in range(1, feature_maps.shape[3] + 1):
	plt.subplot(8, 8, i)
	plt.imshow(feature_maps[0, :, :, i - 1], cmap='gray')
plt.savefig(FEATUREMAP_PATH)
plt.show()
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

# create a new figure for the accuracies
N = stopped_epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, stopped_epoch), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, stopped_epoch), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(LOSS_PATH)
print("[INFO] saved loss Figure")
plt.close()
# #create a new figure for the accuracies
N = stopped_epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, stopped_epoch), H.history["accuracy"], label="Accuracy")
plt.plot(np.arange(0, stopped_epoch), H.history["val_accuracy"], label="val_Accuracy")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(ACCURACY_PATH)
print("[INFO] saved accuracy Figure")
plt.close()

