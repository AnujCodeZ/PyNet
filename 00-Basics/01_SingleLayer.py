import numpy as np

class Layer:

    # Initialization
    def __init__(self, layers):

        """
        Arguments:
        - layers: containing list of layer units(input, hidden, output)
        """
        self.num_classes = layers[-1]
        self.W1 = np.random.randn(layers[0], layers[1]) * 0.01
        self.W2 = np.random.randn(layers[1], layers[2]) * 0.01
        self.b1 = np.zeros((1, layers[1]))
        self.b2 = np.zeros((1, layers[2]))
        self.dW1 = np.zeros((layers[0], layers[1]))
        self.dW2 = np.zeros((layers[1], layers[2]))
        self.db1 = np.zeros((1, layers[1]))
        self.db2 = np.zeros((1, layers[2]))
    
    # Activation functions
    def sigmoid(self, Z):

        return 1. / (1. + np.exp(-Z))
    
    def softmax(self, Z):

        return np.exp(Z) / np.sum(np.exp(Z))

    # Forward pass
    def forward(self, X):

        """
        Arguments:
        - X: features of shape(num_examples, num_features)

        Returns:
        - cache: dictionary containing activations
        """

        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        if self.num_classes == 1:

            A2 = self.sigmoid(Z2)

        else:

            A2 = self.softmax(Z2)

        cache = {
            "A1": A1,
            "A2": A2
        }

        return cache
    
    # Cost function
    def compute_cost(self, A2, Y):

        """
        Arguments:
        - A2: activations of output layer of shape(num_example, num_classes)
        - Y: labels of shape(num_examples, num_classes)

        Returns:
        - loss: cross-entropy loss
        """

        loss = -1 * np.sum(np.dot(np.transpose(Y), np.log(A2)) + \
                           np.dot(np.transpose(1-Y), np.log(1-A2)))
        
        loss = loss / Y.shape[0]

        return loss
    
    # Backward pass
    def backward(self, X, Y, cache):

        """
        Arguments:
        - X: features of shape(num_examples, num_features)
        - Y: labels of shape(num_examples, num_classes)
        - cache: dictionary containing activations
        """

        A1 = cache["A1"]
        A2 = cache["A2"]
        m = X.shape[0]

        dZ2 = A2 - Y
        self.dW2 = np.dot(dZ2, A1.T) / m
        self.db1 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(np.dot(self.W2, dZ2), (1 - A1 ** 2))
        self.dW1 = np.dot(dZ1, X.T) / m
        self.db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    # Update parameters
    def update_params(self, lr):
        
        """
        Arguments:
        - lr: learning rate
        """

        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
    
    # Train
    def train(self, X, Y, epochs, lr):

        """
        Arguments:
        - X: features of shape(num_examples, num_features)
        - Y: labels of shape(num_examples, num_classes)
        - epochs: number of iterations
        - lr: learning rate
        
        Returns:
        - cost: list of losses of per 100th epoch
        """

        cost = []
        for e in range(epochs):

            cache = self.forward(X)

            loss = self.compute_cost(cache["A2"], Y)
            if e % 100 == 0:
                cost.append(loss)
                print("Cost after {}th iteration: {:.3f}".format(e, loss))
            
            self.backward(X, Y, cache)

            self.update_params(lr)

        return cost
