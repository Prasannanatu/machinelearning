import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE
_alpha = 0.9
_epoch = 10
#chaning the above values to get different results#

W_trained = train(Xtrain,Ytrain,_alpha ,_epoch)
Y_train_pred_ =compute_yhat(Xtrain, W_trained)
Loss_train = compute_L(Y_train_pred_, Ytrain)


Y_test_pred_ = compute_yhat(Xtest, W_trained)
Loss_test  = compute_L(Y_test_pred_, Ytest)

print("Y_train_pred_",format(Loss_train,".3f"))
print("Y_test_pred_",format(Loss_test,".3f"))
print("aplha:and epoch are:", _alpha, _epoch)

# from matplotlib import pyplot as plt


# alpha_L = [0.001,0.005,0.01,0.05,0.1,]
# epoch_L =  [1,15000,10]
# train_loss_list =[]
# test_loss_list =[]

# for i in alpha_L:
#     for j in epoch_L:
#         W_trained = train(Xtrain,Ytrain,i ,j)
#         # mtrx = np.zeros((i,j))
#         Y_train_pred_ =compute_yhat(Xtrain, W_trained)
#         train_loss_list[j] = compute_L(Y_train_pred_, Ytrain)

#         Y_test_pred_ = compute_yhat(Xtest, W_trained)
#         test_loss_list[j] = compute_L(Y_test_pred_, Ytest)

        













#########################################

