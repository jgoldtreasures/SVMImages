from libsvm.svmutil import *
from libsvm.commonutil import *


def q1():
    y, x = svm_read_problem("D:\CSDS435_HW6_DogsVsCats\DogsVsCats.train")

    # 10-Fold CV
    svm_train(y, x, '-t 0 -v 10 -q')  # linear

    svm_train(y, x, '-t 1 -d 5 -v 10 -q')  # polynomial

    h_lin = svm_train(y, x, '-t 0 -q')  # linear
    svm_predict(y, x, h_lin)

    h_poly = svm_train(y, x, '-t 1 -d 5 -q')  # polynomial
    svm_predict(y, x, h_poly)

    # Testing
    ytest, xtest = svm_read_problem("D:\CSDS435_HW6_DogsVsCats\DogsVsCats.test")

    svm_predict(ytest, xtest, h_lin)
    svm_predict(ytest, xtest, h_poly)


def main():
    q1()


main()
