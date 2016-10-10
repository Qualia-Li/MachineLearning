import sklearn.metrics as metrics
import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.io as io
import math
import csv
import matplotlib.pyplot as plt

NUM_CLASSES = 10

def load_dataset():
    data = io.loadmat('./data/spam.mat')
    X_train = data['Xtrain']
    labels_train = data['ytrain']
    X_test = data['Xtest']
    labels_test = []

    X_data = np.hstack((X_train, labels_train))
    X_data = np.random.permutation(X_data)

    X_train = X_data[:, 0:-1]
    labels_train = X_data[:, -1].astype(int)
    labels_train.shape = (labels_train.shape[0], 1)
    # print(X_train.shape)
    # print(labels_train.shape)

    print('load_dataset done...')

    return (X_train, labels_train), (X_test, labels_test)


def standardize(X):
    mean = np.mean(X, axis=0)
    mean = np.tile(mean, (X.shape[0], 1))
    std = np.std(X, axis=0)
    std = np.tile(std, (X.shape[0], 1))
    # print(mean.shape)
    # print(istd.shape)

    Xstd = np.divide((X - mean), std)

    XstdMean = np.mean(Xstd, axis=0)
    XstdStd = np.std(Xstd, axis=0)

    # print(XstdMean)
    # print(XstdStd)

    print('standardize done...')

    return Xstd


def transform(X):
    Xtrans = np.log(X + 0.1)

    print('transform done...')

    return Xtrans


def binarize(X):
    Xbin = np.empty([X.shape[0], X.shape[1]])

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] > 0:
                Xbin[i, j] = 1
            else:
                Xbin[i, j] = 0

    print('binarize done...')

    return Xbin


def train_gd(x_train, y_train, alpha, reg, num_iter):
    ''' Build a model from X_train -> y_train using batch gradient descent '''

    return beta, error


def train_sgd(x_train, y_train, alpha, reg, num_iter):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''

    return beta, error


def one_hot(labels):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    encMat = np.eye(NUM_CLASSES)[labels]
    print('one_hot done...')
    return encMat


def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    predictionMat = X.dot(model)
    predictionVec = np.zeros((X.shape[0], 1))

    for i in range(X.shape[0]):
        predictionVec[i] = predictionMat[i][:].argmax()

    # print(predictionVec)
    return predictionVec


def genRand(X):
    # Best result for closed form use
    # p = 5000
    # var = 0.02

    p = 1500
    var = 0.01
    mean = 0.0
    G = np.random.normal(mean, math.sqrt(var), (X.shape[1], p))
    b = np.random.uniform(0.0, 2 * math.pi, (1, p))

    return p, G, b


def phi(X, p, G, b):
    ''' Featurize the inputs using random Fourier features '''
    B = np.tile(b, (X.shape[0], 1))
    # Xlifted = math.sqrt(2.0/p)*np.cos(X.dot(G)+B)
    Xlifted = np.cos(X.dot(G) + B)

    print('lifiting done...')
    return Xlifted


if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()

    X_train_std = standardize(X_train)
    X_train_trans = transform(X_train)
    X_train_bin = binarize(X_train)

    # model, error_gd = train_gd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=50000)
    # pred_labels_train = predict(model, X_train)
    # # pred_labels_test = predict(model, X_test)
    # print("Batch gradient descent")
    # print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    # # print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    # model, error_sgd = train_sgd(X_train, y_train, alpha=1e-5, reg=0.1, num_iter=50000)
    # pred_labels_train = predict(model, X_train)
    # # pred_labels_test = predict(model, X_test)
    # print("Stochastic gradient descent")
    # print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    # # print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    write = 0

    if write == 1:
        results = open('test_results.csv', 'w')
        results_writer = csv.writer(results)
        results_writer.writerow(('Id', 'Category'))
        for i in range(pred_labels_test.shape[0]):
            results_writer.writerow((i, int(pred_labels_test[i])))
        print("Write to file complete...")

    plot = 0

    if plot == 1:
        fig = plt.figure()
        plt.plot(error_gd[:, 0], error_gd[:, 1], 'b', label='GD')
        plt.grid()
        plt.title("Error vs. Iterations for Gradient Descent")
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.legend()

        fig = plt.figure()
        plt.plot(error_sgd[:, 0], error_sgd[:, 1], 'r', label='SGD')
        plt.grid()
        plt.title("Error vs. Iterations for Stochastic Gradient Descent")
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.legend()

        plt.draw()
        plt.show()