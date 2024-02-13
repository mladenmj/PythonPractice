import numpy as np
import pvml
import os

def one_vs_rest_svm_train(X, Y, lambda_, lr=1e-3, steps=1000,
						  init_w=None, init_b=None):
	k = Y.max() + 1
	m, n = X.shape
	W = np.zeros((n, k))
	b = np.zeros(k)
	for c in range(k):
		Ybin = (Y == c)
		w1 = (None if init_w is None else init_w[:, c])
		b1 = (0 if init_b is None else init_b[c])
		Wbin, bbin = pvml.svm.svm_train(X, Ybin, lambda_, lr=lr, steps=steps,
							   init_w=w1, init_b=b1)
		W[:, c] = Wbin
		b[c] = bbin
	return W, b

def one_vs_rest_svm_inference(X, W, b):
	logits = X @ W + b.T
	labels = np.argmax(logits, 1)
	return labels, logits

def one_vs_rest_ksvm_inference(X, Xtrain, alpha, b, kfun, kparam):
	K = pvml.ksvm.kernel(X, Xtrain, kfun, kparam)
	logits = K @ alpha + b
	labels = np.argmax(logits, 1)
	return labels, logits


def one_vs_rest_ksvm_train(X, Y, kfun, kparam, lambda_, lr=1e-3, steps=1000,
						   init_alpha=None, init_b=None):
	k = Y.max() + 1
	m, n = X.shape
	alpha = np.zeros((m, k))
	b = np.zeros(k)
	for c in range(k):
		Ybin = (Y == c)
		a1 = (None if init_alpha is None else init_alpha[:, c])
		b1 = (0 if init_b is None else init_b[c])
		abin, bbin = pvml.ksvm.ksvm_train(X, Ybin, kfun, kparam, lambda_, lr=lr,
								steps=steps, init_alpha=a1, init_b=b1)
		alpha[:, c] = abin
		b[c] = bbin
	return alpha, b

def load_data(filename):
	data = np.loadtxt(filename)

	X = data[:, :-1]
	Y = data[:, -1].astype(int)

	return X, Y

##################################################################################

X, Y = load_data("train.txt.gz")
Xtest, Ytest = load_data("test.txt.gz")
Xval, Yval = load_data("val.txt.gz")

# Gridsearch for one-vs-rest using SVM:
lr = [0.003, 0.01, 0.03]
lambda_ = [0.01, 0.03, 0.05]

for i in range(len(lr)):
    for j in range(len(lambda_)):
        W, b = one_vs_rest_svm_train(X, Y, lambda_[j], lr=lr[i], steps=10000)
        labels, logits = one_vs_rest_svm_inference(X, W, b)
        accuracy = (labels == Y).mean() * 100
        print("\n-> Training accuracy: ", accuracy, "% - LR:", lr[i], " - lambda:", lambda_[j])

        labels, logits = one_vs_rest_svm_inference(Xtest, W, b)
        test_acc = (labels == Ytest).mean() * 100
        print("Test accuracy:", test_acc, "% - LR:", lr[i], " - lambda:", lambda_[j])
        labels, logits = one_vs_rest_svm_inference(Xval, W, b)
        val_acc = (labels == Yval).mean() * 100
        print("Val accuracy:", val_acc, "% - LR:", lr[i], " - lambda:", lambda_[j])

os.system("echo -ne '\007'")    # beep when finished

# Kernel SVM:
print("Kernel SVM:\n")
#kfun = "polynomial"
kfun = "rbf"

# Gridsearch for one-vs-rest using kernel SVM:
kparam = [300]
lambda_ = [0.03, 0.1]
lr = [0.01, 0.03]

for i in range(len(kparam)):
    for j in range(len(lambda_)):
        for k in range(len(lr)):
            alpha, b = one_vs_rest_ksvm_train(X, Y, kfun, kparam[i], lambda_[j], lr[k], steps=2500)
            labels, logits = one_vs_rest_ksvm_inference(X, X, alpha, b, kfun, kparam[i])
            accuracy = (labels == Y).mean() * 100
            print("\n* Training accuracy: ", accuracy, "% - kparam:", kparam[i], " - lambda:", lambda_[j], "LR: ", lr[k])
            labels, logits = one_vs_rest_ksvm_inference(Xtest, X, alpha, b, kfun, kparam[i])
            test_acc = (labels == Ytest).mean() * 100
            print("Test accuracy:", test_acc, "% - kparam:", kparam[i], " - lambda:", lambda_[j], "LR: ", lr[k])
            labels, logits = one_vs_rest_ksvm_inference(Xval, X, alpha, b, kfun, kparam[i])
            val_acc = (labels == Yval).mean() * 100
            print("Val accuracy:", val_acc, "% - kparam:", kparam[i], " - lambda:", lambda_[j], "LR: ", lr[k])


os.system("echo -ne '\007'")    # beep when finished