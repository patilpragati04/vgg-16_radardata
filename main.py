# import the necessary packages
from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import img_to_array
from keras.utils import load_img
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file
BASE_PATH = "dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "JPEGImages"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations.csv"])

# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])


# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32
# load the contents of the CSV annotations file
print("[INFO] loading dataset...")
rows = open(ANNOTS_PATH).read().strip().split("\n")
# initialize the list of data (images), our target output predictions
# (bounding box coordinates), along with the filenames of the
# individual images
data = []
filenames = []
imagePaths = []
labels = []
bboxes = []
# loop over all CSV files in the annotations directory
# loop over the rows
for row in rows:
	# break the row into the filename, bounding box coordinates,
	# and class label
	row = row.split(",")
	(name, width, height, label, xtl, ytl, xbr, ybr, z_order) = row
	# derive the path to the input image, load the image (in OpenCV
	# format), and grab its dimensions
	imagePath = os.path.sep.join([IMAGES_PATH, name])
	#print(imagePath)
	image = cv2.imread(imagePath)
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
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	# update our list of data, targets, and filenames
	data.append(image)
	labels.append(label)
	bboxes.append((xtl, ytl, xbr, ybr))
	filenames.append(name)
	imagePaths.append(imagePath)
# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
# if xtl != "":
# 	bboxes = np.array(bboxes, dtype="float32")
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)
#print(data)

# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data, bboxes, filenames, test_size=0.10,
	random_state=42)
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# only there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well
if len(lb.classes_) == 2:
	labels = to_categorical(labels)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainBBoxes, testBBoxes) = split[2:4]
(trainFilenames, testFilenames) = split[4:]


# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
print("[INFO] saving testing filenames...")
f = open(TEST_FILENAMES, "w")
print(TEST_FILENAMES)
f.write("\n".join(TEST_FILENAMES))
f.close()

# load the VGG16 network, ensuring the head FC layers are left off
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
# softmaxHead = Dense(512, activation="relu")(flatten)
# softmaxHead = Dropout(0.5)(softmaxHead)
# softmaxHead = Dense(512, activation="relu")(softmaxHead)
# softmaxHead = Dropout(0.5)(softmaxHead)
# softmaxHead = Dense(len(lb.classes_), activation="sigmoid",
# 	name="class_label")(softmaxHead)
# put together our model which accept an input image and then output
# bounding box coordinates and a class label
model = Model(
	inputs=vgg.input,
	outputs=(bboxHead))
# define a dictionary to set the loss methods -- categorical
# cross-entropy for the class label head and mean absolute error
# for the bounding box head
# losses = {
# 	"class_label": "binary_crossentropy",
# 	"bounding_box": "mean_squared_error",
# }
#
# # define a dictionary that specifies the weights per loss (both the
# # class label and bounding box outputs will receive equal weight)
# lossWeights = {
# 	"class_label": 1.0,
# 	"bounding_box": 1.0
# }
#
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=INIT_LR)
model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])
print(model.summary())

# # construct a dictionary for our target training outputs
# trainTargets = {
# 	"class_names": trainFilenames,
# 	"bounding_box": trainBBoxes
# }
#
# # construct a second dictionary, this one for our target testing
# # outputs
# testTargets = {
# 	"class_name": testFilenames,
# 	"bounding_box": testBBoxes
# }
# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")
print("[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, trainBBoxes,
	validation_data=(testImages, testBBoxes),
	batch_size=BATCH_SIZE,
	epochs=NUM_EPOCHS,
	verbose=1)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(MODEL_PATH, save_format="h5")

# # serialize the label binarizer to disk
# print("[INFO] saving label binarizer...")
# f = open(config.LB_PATH, "wb")
# f.write(pickle.dumps(lb))
# f.close()
# plot the total loss, label loss, and bounding box loss
lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(N, H.history[l], label=l)
	ax[i].plot(N, H.history["val_" + l], label="val_" + l)
	ax[i].legend()

# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plotPath = os.path.sep.join([PLOT_PATH, "losses.png"])
plt.savefig(plotPath)
plt.close()
# create a new figure for the accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"],
	label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"],
	label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

# save the accuracies plot
plotPath = os.path.sep.join([PLOT_PATH, "accs.png"])
plt.savefig(plotPath)

