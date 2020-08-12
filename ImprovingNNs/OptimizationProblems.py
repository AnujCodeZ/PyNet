import numpy as np
from ..Basics.DeepLayers import DeepNet
from Utils import *

class GradProblems(DeepNet):

    def __init__(self):
        super().__init__()

        self.he_initialization()
    
    """He Weight Initialization
    To overcome vanishing/exploding gradients"""
    def he_initialization(self):

        for i in range(len(self.layers)):
            parameters['W'+str(i+1)] = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2 / self.layers[i])
            parameters['b'+str(i+1)] = np.zeros((1, self.layers[i+1]))

    """Gradient checking
    To check whether our gradient is right or not"""

    # Custom forward for gradient check
    def forward_with_theta(self, X, theta):

        params = vector_to_dictionary(theta, self.layers)
        A = X

        for i in range(len(self.layers - 1)):

            A_prev = A

            Z = np.dot(A_prev, params['W'+str(i+1)]) + params['b'+str(i+1)]
            A = self.relu(Z)
            self.caches.append(A)
        
        Z = np.dot(A_prev, params['W'+str(len(self.layers))]) + params['b'+str(len(self.layers))]
        AL = self.softmax(Z)

        return AL

    def grad_check(self, X, Y, EPSILON = 10e-7):

        # Rolling parameters and gradients
        params = dictionary_to_vector(self.parameters)
        grads = gradients_to_vector(self.gradients)

        num_parameters = len(params)
        gradsApprox = np.zeros((num_parameters, 1))

        for i in range(num_parameters):

            # Cost for theta+EPSILON
            thetaPlus = np.copy(params)
            thetaPlus[i] = thetaPlus[i] + EPSILON

            AL = self.forward_with_theta(X, thetaPlus)
            J_plus = self.compute_cost(AL, Y)

            # Cost for theta+EPSILON
            thetaMinus = np.copy(params)
            thetaMinus[i][0] = thetaMinus[i][0] - EPSILON

            AL = self.forward_with_theta(X, thetaMinus)
            J_minus = self.compute_cost(AL, Y)

            # Compute numerical gradients
            gradsApprox[i] = (J_plus - J_minus) / (2 * EPSILON)

        # Compute difference of numerical and analitical gradients
        numerator = np.linalg.norm(grads - gradsApprox)
        denomenator = np.linalg.norm(grads) + np.linalg.norm(gradsApprox)

        difference = numerator / denomenator

        if difference > 10e-7:

            print("There is a mistake in back-propagation.\n")
        
        else:

            print("Back-propagation is fine.\n")