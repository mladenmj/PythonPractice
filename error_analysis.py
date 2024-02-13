import pvml
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filename):
	data = np.loadtxt(filename)

	X = data[:, :-1]
	Y = data[:, -1].astype(int)

	return X, Y

def confmat_plot(cm):
	x = np.arange(1,cm.shape[0]+1)
	y = np.arange(1,cm.shape[1]+1)

	# Plotting:
	fig, ax = plt.subplots()
	im = ax.imshow(cm, cmap="jet")

	# Show all ticks and labels:
	ax.set_xticks(np.arange(len(x)))
	ax.set_yticks(np.arange(len(y)))
	ax.set_xticklabels(x)
	ax.set_yticklabels(y)

	# Colorbar:
	cbar = ax.figure.colorbar(im, ax=ax)

	# Set ticks position
	ax.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)

	# Turn spines off and create white grid
	for edge, spine in ax.spines.items():
		spine.set_visible(False)

	ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
	ax.tick_params(which="minor", bottom=False, left=False)

	# Print data inside the heatmap-matrix
	for i in range(len(x)):
	    for j in range(len(y)):
	        text = ax.text(j, i, cm[i, j].round(3),
	                       ha="center", va="center", color="w")

	ax.set_title("Confusion matrix\n", fontweight='bold')
	fig.tight_layout()

def confmat(labels, Ytest):
	k = np.max(Ytest) + 1
	cm = np.zeros((k, k))

	for lbl, y in zip(labels, Ytest):
		cm[lbl, y] += 1

	cm /= cm.sum(1, keepdims=True)

	# Rounding numbers to 3 decimal positions:
	tmp_rnd = cm.round(3)

	for i in range(cm.shape[0]):
		if(tmp_rnd.sum(1)[i] != 1):
			tmp_rnd[i, i] += 1-tmp_rnd.sum(1)[i]

	cm = tmp_rnd * 100

	confmat_plot(cm)

	return cm

def mispredict(Xtest, Ytest, probs, labels, img_classes, img_names):
	m = Xtest.shape[0]
	
	pcorrect = probs[np.arange(m), Ytest]
	idx = pcorrect.argsort()

	print("\n* Mispredictions *")
	print("\n* Filename - Probability estimate - Actual class - Predicted class:")

	for i in range(15):
		print(img_names[idx[i]], "-", pcorrect[idx[i]], "-", img_classes[Ytest[idx[i]]], "-", img_classes[labels[idx[i]]])

####################################################

# Data and net loading:
X, Y = load_data("train.txt.gz")
Xtest, Ytest = load_data("test.txt.gz")
Xval, Yval = load_data("val.txt.gz")

predictions = np.load("predictions.npy")
probs = np.load("probs.npy")

# Confusion matrix:
cm = confmat(predictions, Ytest)
# np.save('confusion_matrix', cm)

# Misclassification analysis
img_classes = range(np.max(Y) + 1)
img_names = []

path = "test/"
image_file = os.listdir(path)

for imgname in image_file:
		img_names.append(imgname)

mispredict(Xtest, Ytest, probs, predictions, img_classes, img_names)

plt.show()
