from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split


class NNActivationFunction:
    def __init__(self):
        """
        class containing a univariate function with a single output
        i.e. R^2 -> R^1
        """
        pass

    def forward(self, x):
        """
        implement the forward function
        """
        raise NotImplementedError
    
    def backward(self, x):
        """
        implement the differential of the function
        """
        raise NotImplementedError
    
class Tanh(NNActivationFunction):
    def forward(self, x):
        return np.tanh(x)
    def backward(self, x):
        return 1-np.tanh(x)**2



class NeuralNetwork:
    def __init__(self, type, layer_dimensions: Tuple, activation_function: NNActivationFunction, learning_rate):
        """
        type must be either "classification" or "regression".
        """
        if type not in ("classification", "regression"): raise ValueError(f'invalid type: {type}')
        self.type = type
        self.activation_function = activation_function
        self.layer_dimensions = layer_dimensions
        self.L = len(self.layer_dimensions)-1
        self.weights = [np.empty((1,1))] + [np.random.rand(layer_dimensions[i], layer_dimensions[i+1])-0.5 for i in range(self.L)]
        self.learning_rate = learning_rate
        self.layer_iter = lambda: range(1,self.L + 1)

        self.z = []
        self.a = []

    def forward_simple(self, X: np.ndarray):
        if len(X.shape) != 2:
            raise ValueError(f'X must be a matrix of shape (NxD), where D is input dimensionality - was shape {X.shape}')
        if X.shape[1] != self.layer_dimensions[0]:
            raise ValueError(f'Input dimensionality of X doesn\' match input dimensionality of network - was shape {X.shape}')
        
        z = X
        for l in list(self.layer_iter())[:-1]:
            z = self.activation_function.forward(z @ self.weights[l])

        if self.type == "classification":
            a = z @ self.weights[self.L]
            expa = np.exp(a)
            z = expa/np.sum(expa, axis = 1)[:,None]

        elif self.type == "regression":
            z = z @ self.weights[self.L]

        return z

    def forward_with_gradient(self, X: np.ndarray):
        if len(X.shape) != 2:
            raise ValueError(f'X must be a matrix of shape (NxD), where D is input dimensionality - was shape {X.shape}')
        if X.shape[1] != self.layer_dimensions[0]:
            raise ValueError(f'Input dimensionality of X doesn\' match input dimensionality of network - was shape {X.shape}')

        
        z = [None]*(self.L+1)
        a = [None]*(self.L+1)
        z[0] = X
        for l in self.layer_iter():
            a[l] = z[l-1] @ self.weights[l]
            assert a[l].shape[1] == self.layer_dimensions[l]
            z[l] = self.activation_function.forward(a[l])
            assert z[l].shape[1] == self.layer_dimensions[l]

        if self.type == "classification":
            expa = np.exp(a[self.L])
            z[self.L] = expa/np.sum(expa, axis = 1)[:,None]
            assert z[self.L].shape[1] == self.layer_dimensions[self.L]

        elif self.type == "regression":
            z[self.L] = a[self.L]
        
        self.N = len(X)
        self.z = z
        self.a = a
        return z[self.L]

    
    def backward(self, y_true, y_pred):
        deltas = [None]*(self.L + 1)
        gradient = [None]*(self.L + 1)


        if self.type == "classification":
            deltas[self.L] = (y_pred - y_true).astype(float)

        elif self.type == "regression":
            raise NotImplementedError('Regression gradient not implemented yet + implement accuracy')

        #compute gradients
        for l in reversed(list(self.layer_iter())[:-1]):
            deltas[l] = self.activation_function.backward(self.a[l]) * (deltas[l+1] @ self.weights[l+1].T)
            assert deltas[l].shape == (self.N, self.layer_dimensions[l])
        
        for l in reversed(self.layer_iter()):
            gradient[l] = np.sum(deltas[l][:,None,:]*self.z[l-1][:,:,None], axis = 0)
            assert gradient[l].shape == self.weights[l].shape

        #update weights
        for l in self.layer_iter():
            self.weights[l] = self.weights[l] - gradient[l]*self.learning_rate

    def train(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, n_epochs, stochastic = 0.5):

        for epoch in range(n_epochs):
            if stochastic != 1.:
                subset_idx = np.random.choice(len(X_train), int(len(X_train)*stochastic), replace = False)
            
            y_pred = self.forward_with_gradient(X_train[subset_idx])

            if not epoch%int(n_epochs/20):

                if self.type == "classification":
                    train_accuracy = np.mean(np.argmax(y_pred, axis = 1) == np.argmax(y_train[subset_idx], axis = 1))
                    y_pred_test = self.forward_simple(X_test)
                    test_accuracy = np.mean(np.argmax(y_pred_test, axis = 1) == np.argmax(y_test, axis = 1))
                
                print("epoch:", epoch, "  train acc:",train_accuracy, "  test acc:", test_accuracy)

                if test_accuracy >= 0.95: break

            self.backward(y_train[subset_idx], y_pred)

###################################################

X = np.load("1_Feedforward_Pen_and_Paper/data/X.npy", allow_pickle=True)
y = np.load("1_Feedforward_Pen_and_Paper/data/y.npy", allow_pickle=True)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5)

N, D = X.shape
n_classes = y.shape[1]


# print(N,D,n_classes)
net = NeuralNetwork("classification", (D,15,14,n_classes), Tanh(), 1e-6)
net.train(X_train, X_test, y_train, y_test, 1000)
