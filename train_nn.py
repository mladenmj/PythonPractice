import numpy as np
import matplotlib.pyplot as plt
import pvml
import os

def load_data(filename):
	data = np.loadtxt(filename)

	X = data[:, :-1]
	Y = data[:, -1].astype(int)

	return X, Y

X, Y = load_data("train.txt.gz")
Xtest, Ytest = load_data("test.txt.gz")
Xval, Yval = load_data("val.txt.gz")

mlp = pvml.MLP([X.shape[1], 128, 6])

batch_size = 40
steps = X.shape[0] // batch_size
lr = 0.01
epochs = 1000

train_accs = []
test_accs = []
val_accs = []
plt.ion()

for epoch in range(epochs):
	mlp.train(X, Y, lr=lr, batch=batch_size, steps=steps)
	predictions, probs = mlp.inference(X)
	train_acc = (predictions == Y).mean() * 100
	train_accs.append(train_acc)

	predictions, probs = mlp.inference(Xval)
	val_acc = (predictions == Yval).mean() * 100
	val_accs.append(val_acc)

	predictions, probs = mlp.inference(Xtest)
	test_acc = (predictions == Ytest).mean() * 100
	test_accs.append(test_acc)

	print(epoch, train_acc, test_acc, val_acc)
	plt.clf()
	plt.plot(train_accs)
	plt.plot(test_accs)
	# plt.plot(val_accs)
	plt.legend(["train", "test"])
	plt.title("MLP - Accuracy [%]")
	plt.pause(0.01)


#mlp.save("dice-mlp.npz")
np.save('predictions', predictions)	# Saved for error analysis
np.save('probs', probs)

plt.ioff()
plt.show()

os.system("echo -ne '\007'")	# beep when finished
