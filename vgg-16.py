import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims
import cv2
import os
import pickle
# import the necessary packages
from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import Adam
from model import preprocess_true_boxes,get_random_data
from keras.utils import img_to_array
from keras.utils import load_img
import keras
from PIL import Image
from keras.metrics import Accuracy, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC, BinaryAccuracy
# from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
LOSS_PATH = os.path.sep.join([BASE_OUTPUT, "loss.png"])
ACCURACY_PATH = os.path.sep.join([BASE_OUTPUT, "acc.png"])
FEATUREMAP_PATH = os.path.sep.join([BASE_OUTPUT, "featuremap.png"])
# TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])
INIT_LR = 1e-4
NUM_EPOCHS = 50
batch_size = 32
train_path = 'info/train.txt'
val_path = 'info/val.txt'
input_shape = 224, 224, 3
hwofimage = 224,224
num_classes =2
# perform one-hot encoding on the labels
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)

# only there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well
# if len(lb.classes_) == 2:
# 	labels = to_categorical(labels)
def data_generator(annotation_lines):
    print("Inside Data Gernerator")
    n = len(annotation_lines)
    # line = annotation_lines.split(",")
    # image = Image.open(line[0])
    # box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    i = 0
    # while True:
    image_data = []
    box_data = []
    # for b in range(batch_size):
    for b in range(n):
        # i %= n
        print(b)
        image, box = get_random_data(annotation_lines[b], hwofimage,random=False)
        # print(image,box)
        #image_data.append(image)
        image_data = np.append(image_data, image)
        # box_data.append(box)
        box_data=np.append(box_data,box)
    #     images, boxes =np.append(image,box)
    # print(len(images))
            # i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
    # print(box_data,"kkkk")
        # y_true = (box_data, input_shape, num_classes)
        # yield
        # yield [image_data, *y_true], np.zeros(batch_size)
    print("got the image values")
    return image_data,box_data
def data_generator_wrap(annotation_lines):
    print("Inside Data Generator Wrap")
    n = len(annotation_lines)
    if n==0 or batch_size<=0:
        return None
    # print(annotation_lines)
    return data_generator(annotation_lines)
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
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense((2), activation="sigmoid",
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
with open(train_path) as f:
	train = f.readlines()
with open(val_path) as f:
	val = f.readlines()
num_train = len(train)
num_val = len(val)
# prediction
print("[INFO] training model...")
print("[INFO] training bounding box regressor...")
H = model.fit(
	data_generator_wrap(train),
	validation_data=data_generator_wrap(val),
	steps_per_epoch=batch_size,
	epochs=NUM_EPOCHS,callbacks=[callback,early_stopping],
	verbose=1)

stopped_epoch = (early_stopping.stopped_epoch+1)
print(stopped_epoch)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(MODEL_PATH, save_format="h5")
# for i in range(len(model.layers)):
# 	layer = model.layers[i]
# 	if 'conv' not in layer.name:
# 		continue
# 	print(i, layer.name, layer.output.shape)
# model = Model(inputs=model.inputs, outputs=model.layers[1].output)
# # expand dimensions so that it represents a single 'sample'
# image = expand_dims(train_image, axis=0)
# # prepare the image (e.g. scale pixel values for the vgg)
# image = preprocess_input(image)
# # get feature map for first hidden layer
# feature_maps = model.predict(image)
# fig = plt.figure(figsize=(20, 15))
# for i in range(1, feature_maps.shape[3] + 1):
# 	plt.subplot(8, 8, i)
# 	plt.imshow(feature_maps[0, :, :, i - 1], cmap='gray')
# plt.savefig(FEATUREMAP_PATH)
# plt.show()
# serialize the label binarizer to disk
# print("[INFO] saving label binarizer...")
# f = open(LB_PATH, "wb")
# f.write(pickle.dumps(lb))
# f.close()
# plot the total loss, label loss, and bounding box loss
lossNames = ["loss"]
#N = np.arange(0, stopped_epoch)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

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

