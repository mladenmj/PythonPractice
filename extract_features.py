import numpy as np
import matplotlib.pyplot as plt
import os
import image_features
from PIL import Image, ImageOps

classes = os.listdir("test")
classes.sort()

def feature_extraction(image, image_features, feature_sel):
	features1 = []
	features2 = []
	features3 = []
	features4 = []

	if(feature_sel[0]==1):
		features1 = image_features.color_histogram(image).reshape(-1)

	if(feature_sel[1]==1):
		features2 = image_features.edge_direction_histogram(image)[:64].reshape(-1)

	if(feature_sel[2]==1):
		features3 = image_features.cooccurrence_matrix(image).reshape(-1)

	if(feature_sel[3]==1):
		features4 = image_features.rgb_cooccurrence_matrix(image).reshape(-1)

	features = np.concatenate((features1, features2, features3, features4))

	return features

def process_directory(path, feature_select):
	all_features = []

	path = path + "/"
	image_file = os.listdir(path)

	for imgname in image_file:
		imgpath = path + imgname
		image = plt.imread(imgpath)
		# rgb_cooccurrence_matrix; cooccurrence_matrix; color_histogram;
		# edge_direction_histogram
		features = feature_extraction(image, image_features, feature_select)
		all_features.append(features)

	X = np.stack(all_features, 0)
	# Assign labels 0,...,5
	Y = np.repeat(np.arange(6), X.shape[0] / 6)

	return X, Y

# Input for selecting which features to apply
print("Select desired low-level features:")
feat_sel = np.zeros(4)

strings = ["Color histogram feature (Y/N) ","Edge direction feature (Y/N) ",
			"Co-occurrence matrix feature (Y/N) ","RGB co-occurrence matrix feature (Y/N) "]

for f in range(4):
	instring = input(strings[f])
	if(instring=="Y" or instring=="y"):
		feat_sel[f] = 1
	else:
		feat_sel[f] = 0

def augment_dir(path, newdir):
	copy = 'cp '
	if not os.path.isdir(newdir):	# copy the image in the new directory
		os.mkdir(newdir)
		copy += str(path) + '/* ' + str(newdir) + '/'
		os.system(copy)
	
	# after copying, apply changes
	for img in os.listdir(path):
		pic = Image.open(newdir + '/' + img)
		# Flip image:
		flip_img_name = str(img)[0:-4] + 'a.png'
		flip_img = ImageOps.flip(pic)
		flip_img.save(newdir + '/' + flip_img_name)
		# Mirror image:
		mirr_imgname = str(img)[0:-4] + 'b.png'
		mirr_img = ImageOps.mirror(pic)
		mirr_img.save(newdir + '/' + mirr_imgname)
		# Rotate image of 90 degrees:
		rot_imgname = str(img)[0:-4] + 'c.png'
		rot_img = pic.rotate(90)
		rot_img.save(newdir + '/' + rot_imgname)


##################################################################

# Data loading:
X, Y = process_directory("train", feat_sel)
#X, Y = process_directory("train_augmented", feat_sel)
print("Train features dimension:", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("train.txt.gz", data)

X, Y = process_directory("test", feat_sel)
print("Test features dimension:", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("test.txt.gz", data)

X, Y = process_directory("validation", feat_sel)
print("Test features dimension:", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("val.txt.gz", data)

print("* Extraction completed!")

if not os.path.isdir("train_augmented"):
	print("Type 'yes' if data augmentation desired:")

	instring = input()

	if instring == "yes":
		augment_dir("train", "train_augmented")
		print("* Training set augmented!")

