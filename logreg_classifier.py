import numpy as np
import pvml
import matplotlib.pyplot as plt
import os

def softmax(z):
	E = np.exp(z - z.max(1, keepdims=True))
	return E / E.sum(1, keepdims=True)

def logreg_inference (X, W, b):
    logits = (X @ W) + b.T
    return softmax(logits)

def cross_entropy(P, H):
	return -(H * np.log(P)).sum(1).mean()

def one_hot_vec(Y, classes):
	"""
	One hot vectors generation
	"""
	m = Y.shape[0]
	H = np.zeros((m, classes))
	H[np.arange(m), Y] = 1
	return H

def logreg_train(X, Y, lr=1e-3, steps=15000, init_w=None, init_b=None):
    m, n = X.shape
    k = Y.max() + 1		# Number of classes

    W = (init_w if init_w is not None else np.zeros((n, k)))
    b = (init_b if init_b is not None else np.zeros(k))

    H = one_hot_vec(Y, k)
    train_accs = []

    plt.ion()
    print("* LR:", lr)

    for step in range(steps):
        P = logreg_inference(X, W, b)
        pred = np.argmax(P, 1)
        accuracy = (pred == Y).mean() * 100
        grad_W = (X.T @ (P - H)) / m
        grad_b = (P - H).mean(0)
        W -= lr * grad_W
        b -= lr * grad_b

        L = cross_entropy(P, H)

        if(step % 100==0):
            print("Step:", step, " - Training accuracy:", accuracy)
            # print(step, L, "Accuracy", accuracy)
            # plt.scatter(step, L)
        
        train_accs.append(accuracy)
        plt.clf()
        plt.title("Training accuracy")
        plt.plot(train_accs)
        plt.pause(0.01)
    return W, b

#################################################################

data = np.loadtxt("train.txt.gz")

X = data[:, :-1]
Y = data[:, -1].astype(int)

lr = 5

W, b = logreg_train(X, Y, lr=lr)
print("Training completed!")

data = np.loadtxt("val.txt.gz")
Xval = data[:, :-1]
Yval = data[:, -1]

P = logreg_inference(Xval, W, b)
pred = np.argmax(P, 1)
accuracy = (pred == Yval).mean()

print("VAL ACCURACY:", accuracy * 100, "%")

data = np.loadtxt("test.txt.gz")
Xtest = data[:, :-1]
Ytest = data[:, -1]

P = logreg_inference(Xtest, W, b)
pred = np.argmax(P, 1)
accuracy = (pred == Ytest).mean()

print("TEST ACCURACY:", accuracy * 100, "%")

plt.ioff()
plt.show()

np.save('weights', W)
np.save('bias', b)

os.system("echo -ne '\007'")    # beep when finished
