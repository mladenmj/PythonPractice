import numpy as np
import pvml

def hgda_train(X, Y, priors=None):
    k = Y.max() + 1
    m, n = X.shape
    means = np.empty((k, n))
    invcovs = np.empty((k, n, n))
    if priors is None:
        priors = np.bincount(Y) / m
    for c in range(k):
        indices = (Y == c).nonzero()[0]
        means[c, :] = X[indices, :].mean(0)
        cov = np.cov(X[indices, :].T)
        invcovs[c, :, :] = np.linalg.pinv(cov)
    return (means, invcovs, priors)

def hgda_inference(X, means, invcovs, priors):
    m, n = X.shape
    k = means.shape[0]
    scores = np.empty((m, k))
    for c in range(k):
        det = np.linalg.det(invcovs[c, :, :])
        diff = X - means[None, c, :]
        q = ((diff @ invcovs[c, :, :]) * diff).sum(1)
        scores[:, c] = 0.5 * q - 0.5 * np.log(det) - np.log(priors[c])
    labels = np.argmin(scores, 1)
    return labels, -scores


def ogda_train(X, Y, priors=None):
    k = Y.max() + 1
    m, n = X.shape
    means = np.empty((k, n))
    cov = np.zeros((n, n))
    if priors is None:
        priors = np.bincount(Y) / m
    for c in range(k):
        indices = (Y == c).nonzero()[0]
        means[c, :] = X[indices, :].mean(0)
        cov += priors[c] * np.cov(X[indices, :].T)
    icov = np.linalg.pinv(cov)
    W = -(icov @ means.T)
    q = ((means @ icov) * means).sum(1)
    b = 0.5 * q - np.log(priors)
    return (W, b)


def ogda_inference(X, W, b):
    scores = X @ W + b.T
    labels = np.argmin(scores, 1)
    return labels, -scores

###########################################################################

data = np.loadtxt("train.txt.gz")
X = data[:, :-1]
Y = data[:, -1].astype(int)

data = np.loadtxt("val.txt.gz")
Xval = data[:, :-1]
Yval = data[:, -1].astype(int)

data = np.loadtxt("test.txt.gz")
Xtest = data[:, :-1]
Ytest = data[:, -1].astype(int)

# Heteroschedastic GDA training:
# means, invcovs, priors = hgda_train(X, Y)
# labels, pred = hgda_inference(X, means, invcovs, priors)
# acc = (labels == Y).mean() * 100
# print("Training accuracy: ", acc, "%")

# labels, pred = hgda_inference(Xtest, means, invcovs, priors)
# test_acc = (labels == Ytest).mean() * 100
# print("Test accuracy: ", acc, "%")

# Omoscedastic GDA training:
W, b = ogda_train(X, Y)

labels, pred = ogda_inference(X, W, b)
train_acc = (labels == Y).mean() * 100

labels, pred = ogda_inference(Xtest, W, b)
test_acc = (labels == Ytest).mean() * 100

labels, pred = ogda_inference(Xval, W, b)
val_acc = (labels == Yval).mean() * 100

print("Training accuracy:", train_acc)
print("Test accuracy:", test_acc)
print("Validation accuracy:", val_acc)
