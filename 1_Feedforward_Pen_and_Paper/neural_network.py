from lib2to3.pgen2 import grammar
from typing import Tuple
import numpy as np

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


class ReLU(NNActivationFunction):
    def __init__(self):
        self.ReLU = np.vectorize(lambda x: max(x,0))
    def forward(self, x):
        return self.ReLU(x)
    def backward(self,x):
        return (x>0).astype(float)

class NeuralNetwork:
    def __init__(self, type, layer_dimensions: Tuple, activation_function: NNActivationFunction, learning_rate):
        """
        Simple FFNN neural network class only linear layers
        Loss functions and activation functions in the last layer are inbuilt and cannot be changed
        and depend on the type of task - classification of regression
        For regression, the activation function in last layer is the idendity and loss is mean squared error.
        For classification, activation fucntion in last layer is softmax and loss is mean cross-entropy.

        Paramters
        ---------
        * type: str in {"classification", "regression"}

                the type of task.

        * layer_dimensions: tuple

                The number of neurons in each layer (including the input layer)

                Example: (784, 2000, 100, 10) for the mnist problem.
            
        * activation_function: NNActivationFunction

                Instance of an activation function class that must have two methods spcified: forward and backward. 

                    forward is the just the function
                    
                    backward is the derivative of the function wrt the input.
        
        * learning_rate: float
                
                you know
        """

        # set instance variables
        if type not in ("classification", "regression"): raise ValueError(f'invalid type: {type}')
        self.type = type
        self.activation_function = activation_function
        self.layer_dimensions = layer_dimensions
        self.learning_rate = learning_rate

        # set L and init weights
        self.L = len(self.layer_dimensions)-1
        self.weights = [np.empty((1,1))] + [(np.random.rand(layer_dimensions[i], layer_dimensions[i+1])-0.5)*0.5 for i in range(self.L)]
        
        # helper: iterator over layers (1,2,3, ... ,L)
        self.layer_iter = lambda: range(1,self.L + 1)

        self.n_parameters = sum([layer_dimensions[l-1]*layer_dimensions[l] for l in self.layer_iter()])

        #init empty z and a.
        self.z = []
        self.a = []

    def forward_simple(self, X: np.ndarray):
        """
        compute simple forward pass without storing values for computing gradient later - otherwise same is forward_with_gradient
        """
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
        """
        Compute the forward pass of the input data points in X and
        return the output while keeping track of forward values to compute gradient later - 
        (only used on a batch of subset (stochastic) of training set)


        Paramters
        ---------
        * X: np.ndarray

                You know
        
        Return
        ----------
        * y_pred: np.ndarray
        
                The output of the network
        
        """

        if len(X.shape) != 2:
            raise ValueError(f'X must be a matrix of shape (NxD), where D is input dimensionality - was shape {X.shape}')
        if X.shape[1] != self.layer_dimensions[0]:
            raise ValueError(f'Input dimensionality of X doesn\'t match input dimensionality of network - was shape {X.shape}')
        
        # init empty list of z and a. 
        # a[l][j] is the linear combination of naurons in layer l-1 with the according weights
        # z[l][j] means the activation of neuron j in layer l, i.e. z[l][j] = h_l(a[l][j])

        z = [None]*(self.L+1)
        a = [None]*(self.L+1)


        #compute forward pass
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
        
        # make instance variables, so they can be retreived from backward function
        self.N = len(X)
        self.z = z
        self.a = a
        return z[self.L]

    
    def backward(self, y_true, y_pred):
        """
        Compute and return the gradient of the loss weights with respect to the parameters (weights) of the network
        (only used on a batch of a subset (stochastic) of training set)

        Paramters
        ---------
        * y_true, y_pred: np ndarray

                You know

        
        Return
        --------
        gradient: list of ndarrays

                The gradient. [gradient wrt weights in layer 'l' for l in (0, 1, 2, ... , L) ]

                Note that l=0 is included but contains nothing as there are no weights in layer 0

                This is just to make the indexing consistent with the litterature (1-indexing)

        
        """

        # init empty lists
        deltas = [None]*(self.L + 1)
        gradient = [None]*(self.L + 1)


        #compute deltas in layer L
        if self.type == "classification":
            deltas[self.L] = (y_pred - y_true).astype(float)

        elif self.type == "regression":
            raise NotImplementedError('Regression gradient not implemented yet + implement accuracy')

        #compute gradients

        #compute deltas in layers l = (L-1, L-2 , ... , 1)
        for l in reversed(list(self.layer_iter())[:-1]):
            deltas[l] = self.activation_function.backward(self.a[l]) * (deltas[l+1] @ self.weights[l+1].T)
            assert deltas[l].shape == (self.N, self.layer_dimensions[l])
        
        #compute gradients in layers l = (L-1, L-2 , ... , 1)
        for l in reversed(self.layer_iter()):
            gradient[l] = np.sum(deltas[l][:,None,:]*self.z[l-1][:,:,None], axis = 0)
            assert gradient[l].shape == self.weights[l].shape

        return gradient

    def train(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, n_epochs,
                    stochastic = 0.5, n_prints = None, batch_size = 1000, goal_accuracy = 0.95):
        """
        train the model using stochastic gradient descent for 'n_epoch' epochs

        Parameters
        ----------
         * X_train, X_test, y_train, y_test: numpy ndarray
                
                You know what it is

         * n_epochs: int
         
                number of epochs to train the model - i.e. number of gradient steps
         
         * stochastic: float in ]0, 1]
                
                The size of the subset of training set on which to compute the gradient.
                To control the randomness of the gradient
        
        * n_prints: int
        
                the number of status prints, (epoch no, train accuracy, test accuracy)
        
        * batch_size: int
                
                the size of the batches the we do forward and backward pass on
        
        * goal_accuracy: float in ]0,1]
        
                the desired accuracy
        
        """
        assert 0 < stochastic <= 1

        N_train = int(len(X_train)*stochastic)
        N_test = len(X_test)

        subset_idx = np.arange(len(X_train))

        for epoch in range(n_epochs):
            if stochastic != 1.:
                subset_idx = np.random.choice(len(X_train), N_train, replace = False)
            
            # init total gradient for subset
            gradient = [np.zeros_like(w) for w in self.weights]

            # loop over train batches to compute total gradient
            i = 0
            batch_corrects = 0
            while i<N_train:

                #compute batch predictions and batch accuracy
                y_pred = self.forward_with_gradient(X_train[subset_idx[i:i+batch_size]])

                if self.type == "classification":
                    batch_corrects += np.sum(np.argmax(y_pred, axis = 1) == np.argmax(y_train[subset_idx[i:i+batch_size]], axis = 1))
                elif self.type == "regression":
                    raise NotImplementedError('regression train accuracy is not implemented')

                #compute the batch gradient and add to the total gradient
                batch_gradient = self.backward(y_train[subset_idx[i:i+batch_size]], y_pred)
                for l in self.layer_iter():
                    gradient[l] += batch_gradient[l]
                
                i += batch_size
            
            train_accuracy = batch_corrects/N_train
            
            # print status (epoch train_acc test_acc) if requirements are satisfied:
            #    n_prints is not given
            #    n_prints is greater than the numper of epochs
            #    or we are at an epoch we want to print accordin to n_prints
            if n_prints == None or n_epochs < n_prints or not epoch%int(n_epochs/n_prints):

                # loop over test batches to compute test accuracy
                batch_corrects = 0
                i = 0
                while i < N_test:

                    #compute batch predictions and batch accuracy
                    y_pred_test = self.forward_simple(X_test[i:i+batch_size])

                    if self.type == "classification":
                        batch_corrects += np.sum(np.argmax(y_pred_test, axis = 1) == np.argmax(y_test[i:i+batch_size], axis = 1))
                    elif self.type == "regression":
                        raise NotImplementedError('regression test accuracy is not implemented')

                    i += batch_size
                
                test_accuracy = batch_corrects/N_test

                # weight_change_step_l2_norm = sum([np.sum(gradient[l]**2) for l in self.layer_iter()])**0.5*self.learning_rate/N_train
		
                avg_weight_change = sum([np.sum(np.abs(gradient[l])) for l in self.layer_iter()])*self.learning_rate/N_train/self.n_parameters 
                avg_weight        = sum([np.sum(np.abs(self.weights[l])) for l in self.layer_iter()])/self.n_parameters 
		

                # print status
                print("epoch:", epoch, (6-len(str(epoch)))*" ", f"train acc: {train_accuracy:.3f}   test acc: {test_accuracy:.3f}   step size: {avg_weight_change:.3}  avg weight: {avg_weight:.3}")

                # break if reached the goal accuracy
                if test_accuracy >= goal_accuracy: break

            # update the weights using the gradient
            for l in self.layer_iter():
                self.weights[l] = self.weights[l] - self.learning_rate*gradient[l]/N_train
                # gradient[l]/N_train is the gradient when using mean loss instead of total loss
                # this is to make the learning rate consistent, and not grow proportionally with sample size
                

