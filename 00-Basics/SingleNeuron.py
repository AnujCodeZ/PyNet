import numpy as np

class Neuron:

    # Initialization
    def __init__(self, num_features):

        """
        Arguments:
        - num_features: number of features in dataset.
        """

        self.weights = np.zeros((num_features, 1))
        self.bias = 0
        self.num_features = num_features
        self.dw = np.zeros((num_features, 1))
        self.db = 0
    
    # Forward pass
    def forward(self, X):

        """
        Arguments:
        - X: features shape(num_examples, num_features)

        Returns:
        - A: activations of shape(num_examples, 1)
        """

        Z = np.dot(X, self.weights) + self.bias
        A = 1.0 / (1.0 + np.exp(-Z)) # sigmoid

        return A
    
    # Cost function
    def compute_cost(self, A, Y):
        
        """
        Arguments:
        - A: activations of shape(num_examples, 1)
        - Y: labels of shape(num_examples, 1)

        Return:
        - loss: cross-entropy loss
        """
        
        loss = -1 * np.sum(np.dot(np.transpose(Y), np.log(A)) + \
                           np.dot(np.transpose(1-Y), np.log(1-A)))
        
        loss = loss / Y.shape[0]

        return loss
    
    # Backward pass
    def backward(self, X, Y, A):

        """
        Arguments:
        - X: features of shape(num_examples, num_features)
        - Y: labels of shape(num_examples, 1)
        - A: activations of shape(num_examples, 1)
        """

        self.dw = np.dot(X.T, (A - Y)) / X.shape[0]
        self.db = np.sum((A - Y), axis=1, keepdims=True) / X.shape[0]
    
    # Update parameters
    def update_params(self, lr):

        """
        Arguments:
        - lr: learning rate
        """

        self.weights -= lr * self.dw
        self.bias -= lr * self.db
    
    # Train
    def train(self, X, Y, epochs, lr):

        """
        Arguments:
        - X: features of shape(num_examples, num_features)
        - Y: labels of shape(num_examples, 1)
        - epochs: number of iterations/epochs
        - lr: learning rate
        
        Returns:
        - cost: list of losses per 100th epoch
        """

        cost = []
        for e in range(epochs):

            A = self.forward(X)

            loss = self.compute_cost(A, Y)
            if e % 100 == 0:
                cost.append(loss)
                print("Cost after {}th iteration: {:.3f}".format(e, loss))

            self.backward(X, Y, A)

            self.update_params(lr)
        
        return cost
    
