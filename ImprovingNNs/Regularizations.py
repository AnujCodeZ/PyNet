import numpy as np
from Utils import *

""" L1 and L2 Regularization
Used to overcome high variance or overfitting.
"""
class Regularizer(DeepNet):
    
    def __init__(self):
        super(DeepNet, self).__init__()
    
    def compute_cost_with_regularization(self, AL, Y, lambd=0.01):
        """Compute L2 regularized cost
        For L1 change np.square(self.parameters) to self.parameters

        Args:
            AL (numpy.ndarray): Activations by forward pass.
            Y (numpy.ndarray): True labels.
            lambd (float, optional): Regularization constant. Defaults to 0.01.

        Returns:
            cost: Regularized cost.
        """
        m = Y.shape[0]
        normal_cost = self.compute_cost(AL, Y)
        regularized_cost = 0
        for i in range(len(self.layers)):
            regularized_cost += np.sum(np.square(self.parameters['W'+str(i+1)]))
        
        cost = normal_cost + regularized_cost
        
        return cost
    
    def backward_with_regularization(self, X, Y, lambd=0.01):
        """Calculates gradients with L2 regularization.
        For L1 remove np.sum(self.parameters['W'+...]) terms.

        Args:
            X (numpy.ndarray): Features.
            Y (numpy.ndarray): Labels.
            lambd (float, optional): Regularization constant. Defaults to 0.01.
        """
        As = {}
        As['A'+str(0)] = X
        m = X.shape[0]

        for i in range(len(self.layers)):
            
            As['A'+str(i+1)] = self.caches[i]
        
        dZL = As['A'+str(len(self.layers))] - Y
        self.gradients['W'+str(len(self.layers))] = (np.dot(dZL, As['A'+str(len(self.layers))].T) + 
                                                     (lambd) * np.sum(self.parameters['W'+str(len(self.layers))])) / m
        self.gradients['b'+str(len(self.layers))] = np.sum(dZL, axis=1, keepdims=True) / m

        dZ = dZL

        for i in reversed(range(len(self.layers - 1))):

            dZ_prev = dZ

            dZ = np.dot(np.dot(self.parameters['W'+str(i+2)], dZ_prev), [1 if As['A'+str(i+1)] > 0 else 0])
            self.gradients['W'+str(i+1)] = (np.dot(dZ, As['A'+str(i)].T) + 
                                            (lambd) * np.sum(self.parameters['W'+str(i+1)])) / m
            self.gradients['b'+str(i+1)] = np.sum(dZ, axis=1, keepdims=True) / m
    
""" Dropout
Shut down few neurons to a probability"""
class Dropout(DeepNet):
    
    def __init__(self, x, keep_prob=0.8):
        """Apply dropout to x.

        Args:
            x (numpy.ndarray): Layer
            keep_prob (float, optional): probability of keeping units. Defaults to 0.8.

        Returns:
            (x, d): Dropped layer and mask of dropping units
        """
        super(DeepNet, self).__init__()
        self.keep_prob = keep_prob
        d = np.random.randn(x.shape)
        d = (d < keep_prob).astype(int)
        x = np.multiply(x, d)
        x = x / keep_prob
        
        return (x, d)
    
    def backward_with_dropout(self, dx, d):
        """Apply dropout to backward of layer x.

        Args:
            dx (numpy.ndarray): backward of layer x.
            d (numpy.ndarray): Layer of 0's and 1's to dropout.

        Returns:
            dx: Dropped backward layer.
        """
        dx = np.multiply(dx, d)
        dx = dx / self.keep_prob 
        
        return dx