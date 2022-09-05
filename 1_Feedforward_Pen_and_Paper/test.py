import importlib
import os
from sklearn.model_selection import train_test_split
import numpy as np

NeuralNetwork = importlib.import_module('neural_network').NeuralNetwork
Tanh = importlib.import_module('neural_network').Tanh


X = np.load("1_Feedforward_Pen_and_Paper/data/X.npy", allow_pickle=True)
y = np.load("1_Feedforward_Pen_and_Paper/data/y.npy", allow_pickle=True)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5)

N, D = X.shape
n_classes = y.shape[1]

#  train(self, X_train, X_test, y_train, y_test, n_epochs,
#       stochastic, n_prints, batch_size, goal_accuracy):


# print(N,D,n_classes)
net = NeuralNetwork("classification", (D,100,100,20,n_classes), Tanh(), 1e-2)
net.train(X_train, X_test, y_train, y_test, 1000)
