# import the necessary packages
from keras.utils import img_to_array
from keras.utils import load_img
from keras.models import load_model
import matplotlib.pyplot as plt
from numpy import expand_dims
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import cv2
import numpy as np
import os
BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector_lr10-7_bs100_ep25.h5"])
image = "dataset/Job2/JPEGImages/11-08-2022_12-36-26-379.jpg"
print("[INFO] loading object detector...")
model = load_model(MODEL_PATH)
print("model is loaded")
for i in range(len(model.layers)):
	layer = model.layers[i]
	if 'conv' not in layer.name:
		continue
	print(i, layer.name, layer.output.shape)
model1 = Model(inputs=model.inputs , outputs=model.layers[1].output)

# prepare the image (e.g. scale pixel values for the vgg)
train_image = load_img(image, target_size=(224, 224))
train_image = img_to_array(train_image)
image = expand_dims(train_image, axis=0)
image /= 255.
# prepare the image (e.g. scale pixel values for the vgg)
image = preprocess_input(image)
layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

layer_outputs = [layer.output for layer in model.layers if layer.name in layer_names]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
intermediate_activations = activation_model.predict(image)

images_per_row = 8
max_images = 8
# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, intermediate_activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]
    n_features = min(n_features, max_images)

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    # Display the grid
    scale = 2. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.axis('off')
    plt.title(layer_name)
    plt.grid(True)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()
# plt.savefig('output/featureMaps', display_grid)


