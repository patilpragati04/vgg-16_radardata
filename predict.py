# import the necessary packages
from keras.utils import img_to_array
from keras.utils import load_img
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from numpy import expand_dims
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# from sklearn.linear_model import LogisticRegressions
# from sklearn.metrics import confusion_matrix
import mimetypes
import argparse
import imutils
import cv2
import os
# # define the base path to the input dataset and then use it to derive
# # the path to the images directory and annotation CSV file
# BASE_PATH = "dataset"
ImageAndGTPATH = "info/IOU.txt"
Test_IMAGES_PATH = "dataset/train_images/10-27-2022 14-29-16-834.jpg"
# # Test_IMAGES_PATH = os.path.sep.join([BASE_PATH, "test_images"])
# print(Test_IMAGES_PATH,"kkkk")
# # Test_ANNOTS_PATH ="image10.xml"
# # Test_ANNOTS_PATH = os.path.sep.join([BASE_PATH, "test_annotations.csv"])
# # define the path to the base output directory
BASE_OUTPUT = "output"
# # define the path to the output serialized model, model training plot,
# # and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
FEATUREMAP_PATH = os.path.sep.join([BASE_OUTPUT, "featuremap.png"])
predictedimage_PATH = os.path.sep.join([BASE_OUTPUT, "predictedimage.png"])

# PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
# # test_rows = open(Test_ANNOTS_PATH).read().strip().split("\n")
# test_data =[]
# testFilenames = []
test_imagepaths=[]
# labels = []
# testbboxes =[]
# name = "image10.jpg"
# label ="Human"
# width = 840
# height = 630
# xtl=424.1
# ytl=182.2
# xbr=547.77
# ybr =206.6
# z_order = 0
# # loop over the rows
# # for test_row in test_rows:
# # 	# break the row into the filename, bounding box coordinates,
# # 	# and class label
# # 	test_row = test_row.split(",")
# # 	(name, width, height, label, xtl, ytl, xbr, ybr, z_order) = test_row
# # 	# derive the path to the input image, load the image (in OpenCV
# # 	# format), and grab its dimensions
# # 	# test_imagepath = os.path.sep.join([Test_IMAGES_PATH, name])
# # 	#print(imagePath)
# # 	image = cv2.imread(Test_IMAGES_PATH)
# # 	# plt.imshow(image)
# # 	# plt.show()
# # 	(h, w) = image.shape[:2]
# # 	# scale the bounding box coordinates relative to the spatial
# # 	# dimensions of the input image
# # 	if xtl != "":
# # 		xtl = float(xtl) / w
# # 		ytl = float(ytl) / h
# # 		xbr = float(xbr) / w
# # 		ybr = float(ybr) / h
# # 	else:
# # 		xtl = 0
# # 		ytl = 0
# # 		xbr = 0
# # 		ybr = 0
# # 		label = "No Human"
# # load the image and preprocess it
# test_image = load_img(Test_IMAGES_PATH, target_size=(224, 224))
# test_image = img_to_array(test_image)
# # update our list of data, targets, and filenames
# test_data.append(test_image)
# labels.append(label)
# testbboxes.append((xtl, ytl, xbr, ybr))
# testFilenames.append(test_image)
test_imagepaths.append(Test_IMAGES_PATH)
# # construct the argument parser and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--input", required=True,
# # 	help="datasets")
# # args = vars(ap.parse_args())
#
# # determine the input file type, but assume that we're working with
# # single input image
# # filetype = mimetypes.guess_type(args["input"])[0]
# # imagePaths = [args["input"]]
# # if the file type is a text file, then we need to process *multiple*
# # images

# load the filenames in our testing file and initialize our list
# of image paths



# loop over the filenames
# for f in test_imagepaths:
# 		# construct the full path to the image filename and then
# 		# update our image paths list
# 		p = os.path.sep.join([Test_IMAGES_PATH, f])
# 		imagePaths.append(p)
# load our trained bounding box regressor from disk
print("[INFO] loading object detector...")
model = load_model(MODEL_PATH)
print("model is loaded")
model1 = Model(inputs=model.inputs , outputs=model.layers[1].output)
ground_truth=[]
# xmin = int(443.79)
# ymin = int(59.56)
# xmax = int(507.92)
# ymax = int(90.43)
# image =load_img(Test_IMAGES_PATH)
# cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
# 			  (0, 0, 255), 2)
# cv2.imshow("Output", image)
# loop over the images that we'll be testing using our bounding box
# regression model
for imagePath in test_imagepaths:
	# load the input image (in Keras format) from disk and preprocess
	# it, scaling the pixel intensities to the range [0, 1]
	print(imagePath)
	image = load_img(imagePath, target_size=(224, 224, 3))
	# cv2.imread(image)
	image = img_to_array(image)
	# image = cv2.imread(image)
	print((image.shape),"jjj")
	image = np.expand_dims(image, axis=0)
	# print(len(image),"jjj")

	# # make bounding box predictions on the input image
	preds = model.predict(image)[0]
	(startX, startY, endX, endY) = preds
	print(preds)
	# (xmin,ymin,xmax,ymax)= ground_truth
	# load the input image (in OpenCV format), resize it such that it
	# fits on our screen, and grab its dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=224)
	(h, w) = image.shape[:2]
	print(w,h,"HERE")
	# scale the predicted bounding box coordinates based on the image
 	# dimensions
	xmin = int(443.79 *(224/840))
	ymin = int(59.56 *(168/630))
	xmax = int(507.92*(224/840))
	ymax = int(90.43*(168/630))

	# draw the predicted bounding box on the image
	# cv2.rectangle(image, (startX, startYstartX = int(startX * w)
	startX = int(startX *  168)
	startY = int(startY* 224)
	endX = int(endX * 168)
	endY = int(endY * 224)
	 # print(startX,startY,endX,endY)), (endX, endY),
		# 		  (0, 255, 0), 2)
	# #draw ground truth of an image
	cv2.rectangle(image,(xmin,ymin),(xmax,ymax),
				   (0,0,255),2)
	cv2.rectangle(image,(startX,startY),(endX,endY),(0, 255, 0), 2)
	# show the output image
	cv2.imshow("Output", image)
	cv2.imwrite(predictedimage_PATH, image)
	cv2.waitKey(0)

	# expand dimensions so that it represents a single 'sample'
	# image = load_img(predictedimage_PATH, target_size=(224, 224))
	# print("VVVVV")
	train_image = load_img(predictedimage_PATH, target_size=(224, 224))
	train_image = img_to_array(train_image)
	image = expand_dims(train_image, axis=0)
	image /= 255.
	# prepare the image (e.g. scale pixel values for the vgg)
	image = preprocess_input(image)
	# feature_maps = model.predict(image)
	# print(len(feature_maps.shape))
	# fig = plt.figure(figsize=(20, 15))
	# for i in range(0, 64):
	# 	plt.subplot(4, 4, i)
	# 	plt.imshow(feature_maps[0, :, :, i - 1], cmap='gray')
	# plt.savefig(FEATUREMAP_PATH)
	# plt.show()
	# calculating features_map
	# features = model1.predict(image)
	#
	# fig = plt.figure(figsize=(20, 15))
	# for i in range(1, features.shape[3] + 1):
	# 	plt.subplot(8, 8, i)
	# 	plt.imshow(features[0, :, :, i - 1], cmap='gray')
	#
	# plt.show()
	# layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
	#
	# layer_outputs = [layer.output for layer in model.layers if layer.name in layer_names]
	# activation_model = Model(inputs=model.input, outputs=layer_outputs)
	# intermediate_activations = activation_model.predict(image)
	#
	# images_per_row = 8
	# max_images = 8
	# Now let's display our feature maps
	# for layer_name, layer_activation in zip(layer_names, intermediate_activations):
	# 	# This is the number of features in the feature map
	# 	n_features = layer_activation.shape[-1]
	# 	n_features = min(n_features, max_images)
	#
	# 	# The feature map has shape (1, size, size, n_features)
	# 	size = layer_activation.shape[1]
	#
	# 	# We will tile the activation channels in this matrix
	# 	n_cols = n_features // images_per_row
	# 	display_grid = np.zeros((size * n_cols, images_per_row * size))
	#
	# 	# We'll tile each filter into this big horizontal grid
	# 	for col in range(n_cols):
	# 		for row in range(images_per_row):
	# 			channel_image = layer_activation[0,
	# 							:, :,
	# 							col * images_per_row + row]
	# 			# Post-process the feature to make it visually palatable
	# 			channel_image -= channel_image.mean()
	# 			channel_image /= channel_image.std()
	# 			channel_image *= 64
	# 			channel_image += 128
	# 			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
	# 			display_grid[col * size: (col + 1) * size,
	# 			row * size: (row + 1) * size] = channel_image
	#
	# 	# Display the grid
	# 	scale = 2. / size
	# 	plt.figure(figsize=(scale * display_grid.shape[1],
	# 						scale * display_grid.shape[0]))
	# 	plt.axis('off')
	# 	plt.title(layer_name)
	# 	plt.grid(True)
	# 	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	# plt.show()
	# plt.savefig('output/featureMaps', display_grid)


