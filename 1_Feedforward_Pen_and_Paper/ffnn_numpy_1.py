from turtle import forward
from typing import Tuple
import numpy as np


class NNActivationFunction:
    def __init__():
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
    def __init__(self, type, layer_dimensions: Tuple, activation_function: NNActivationFunction):
        """
        type must be either "classification" or "regression".
        """
        if type not in ("classification", "regression"): raise ValueError(f'invalid type: {type}')
        self.type = type
        self.activation_function = activation_function
        self.layer_dimensions = layer_dimensions
        self.L = len(self.layer_dimensions)-1
        self.weights = [np.empty(1,1)] + [np.random.rand(layer_dimensions[i], layer_dimensions[i+1]) for i in range(self.L)]

        self.layer_iter = lambda: range(1,self.L + 1)

        self.z = []
        self.a = []

    def forward(self, X):
        z = [None]*(self.L+1)
        a = [None]*(self.L+1)
        z[0] = X
        for l in self.layer_iter():
            a[l] = z[l-1] @ self.weights[l]
            z[l] = self.activation_function.forward(a[l])
        if self.type == "classification":
            expa = np.exp(a[self.L])
            z[self.L] = expa/np.sum(expa, axis = 1)
            return z[self.L]
        elif self.type == "regression":
            z[self.L] = a[self.L]
        
        self.z = z
        self.a = a
        return z[self.L]

    
    def backward(self, y_true, y_pred):
        deltas = [None]*(self.L + 1)
        gradient = [None]*(self.L + 1)

        if self.type == "classification":
            deltas[self.L] = y_pred - y_true

        elif self.type == "regression":
            raise NotImplementedError('Regression gradient not implemented yet')

        for l in reversed(list(self.layer_iter())[1:]):
            deltas[l] = self.activation_function.backward(self.a[l]) * deltas[l+1] @ self.weights[l].T
            gradient = None #write the derivative wrt the weights here
            # then the training loop needs to be implemented



            

        
