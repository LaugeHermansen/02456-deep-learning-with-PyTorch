print("Running ffnn_numpy_mnist.py")

print("loading libraries ... ")
import importlib
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

try:
    os.chdir("C:/Users/Bruger/Documents/02456-deep-learning-with-PyTorch")
except:
    try:
        os.chdir("D:/DTU/02456-deep-learning-with-PyTorch")
    except:
        raise FileNotFoundError("Couldn't find directory")
        

NeuralNetwork = importlib.import_module('neural_network').NeuralNetwork
Tanh = importlib.import_module('neural_network').Tanh
ReLU = importlib.import_module('neural_network').ReLU

print("Loading and ecoding data ... ")

data = pd.read_csv("1_Feedforward_Pen_and_Paper/data/MNIST.csv", dtype = float).to_numpy()

# data = data[np.random.choice(len(data), 5000, replace = False)]

encoder = OneHotEncoder()
y = encoder.fit_transform(data[:,0:1]).toarray()
X = data[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.01)

N, D = X.shape
n_classes = y.shape[1]


print(f"number of samples: {N}, input dimension: {D}, number of classes: {n_classes}")

print("Initializing and training network")


# net = NeuralNetwork("classification", (D,300,70,20,15,12,n_classes), Tanh(), 5e-1)
net = NeuralNetwork("classification", (D,400,200,100,50,25,17,15,13,11, n_classes), Tanh(), 3e-1)


# net = NeuralNetwork("classification", (D,300,70,20,15,12,n_classes), ReLU(), 1e-5)
# net = NeuralNetwork("classification", (D,300,70,20,15,12,n_classes), ReLU(), 5e-1)


net.train(X_train, X_test, y_train, y_test, n_epochs = 10000, n_prints = None, stochastic = 0.15, batch_size = 2000, goal_accuracy = 1.)

#  stochastic = 0.5, n_prints = None, batch_size = 100, goal_accuracy = 0.95

# https://fb.watch/flUMgUbycO/
