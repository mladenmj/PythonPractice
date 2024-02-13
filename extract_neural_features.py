import numpy as np
import matplotlib.pyplot as plt
import os
import pvml

def convert_to3D(image):
    width, height = image.shape
    out = np.empty((width, height, 3), dtype=np.uint8)
    out[:, :, 0] = image
    out[:, :, 1] = image
    out[:, :, 2] = image
    return out

def extract_neural_features(im, net):
	im = convert_to3D(im)
	print("DEBUG:", im[None, :, :, :].shape)
	activations = net.forward(im[None, :, :, :])
	features = activations[-3]
	features = features.reshape(-1)
	return features

def process_directory(path, net):
	all_features = []
	all_labels = []

	path = path + "/"
	image_file = os.listdir(path)

	for imgname in image_file:
		imgpath = path + imgname
		image = plt.imread(imgpath)
		features = extract_neural_features(image, net)
		features = features.reshape(-1)
		all_features.append(features)

	X = np.stack(all_features, 0)
	# Assign labels 0,...,5
	Y = np.repeat(np.arange(6), X.shape[0] / 6)

	return X, Y

cnn = pvml.PVMLNet.load("pvmlnet.npz")

X, Y = process_directory("test", cnn)
print(X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("testNN.txt.gz", data)

X, Y = process_directory("train", cnn)
print(X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("trainNN.txt.gz", data)

print("Extraction completed!")