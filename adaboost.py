from libsvm.svmutil import *
from libsvm.commonutil import *
import math
import numpy as np
import sklearn.svm as svm
import scipy.sparse
import pandas as pd


# I could not get the weighted version to work for LIBSVM so I am using sklearn for that
from sklearn.metrics import confusion_matrix, accuracy_score


def adaboost(K=10):  # linear had higher test accuracy
    y, x = svm_read_problem("D:\CSDS435_HW6_DogsVsCats\DogsVsCats.train", return_scipy=True)
    yt, xt = svm_read_problem("D:\CSDS435_HW6_DogsVsCats\DogsVsCats.test", return_scipy=True)
    x = scipy.sparse.csr_matrix.toarray(x)
    m = len(y)
    w = [1 / m] * m
    w = np.array(w)
    hes = [None] * K
    alphas = [None] * K
    for t in range(K):
        h = svm.LinearSVC()
        h.fit(x, y, w)
        hes[t] = h
        y_new = h.predict(x)
        error = 0
        for i in range(m):
            error += w[i] if y[i] != y_new[i] else 0
        alpha = 0.5 * math.log((1 - error) / error)
        alphas[t] = alpha

        individ = np.zeros(m)
        for i in range(m):
            individ[i] = w[i] * math.exp(-1 * alpha * y[i] * y_new[i])
        z = sum(individ)
        w = individ / z
    H = np.zeros(m)
    for t in range(K):
        y_new = hes[t].predict(x)
        H += alphas[t] * np.array(y_new)
    H = np.sign(H)
    print(accuracy_score(y, H))

    Ht = np.zeros(m)
    for t in range(K):
        y_new = hes[t].predict(xt)
        Ht += alphas[t] * np.array(y_new)
    Ht = np.sign(Ht)
    print(accuracy_score(yt, Ht))


def pretty_print(matrix):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(pd.DataFrame(matrix))


def main():
    # adaboost(10)
    adaboost(20)


main()
