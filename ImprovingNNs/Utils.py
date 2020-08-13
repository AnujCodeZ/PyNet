import numpy as np

class DeepNet:

    # Initialization
    def __init__(self, layers):

        self.layers = layers
        self.parameters = {}
        self.gradients = {}
        self.caches = []

        self.initialize()
    
    def initialize(self):

        for i in range(len(self.layers)):

            self.parameters['W'+str(i+1)] = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
            self.parameters['b'+str(i+1)] = np.zeros((1, self.layers[i+1]))
    
    # Activations
    def relu(self, Z):

        return max(0, Z)
    
    def softmax(self, Z):

        return np.exp(Z) / np.sum(np.exp(Z))
    
    # Forward Pass
    def forward(self, X):

        A = X

        for i in range(len(self.layers - 1)):

            A_prev = A

            Z = np.dot(A_prev, self.parameters['W'+str(i+1)]) + self.parameters['b'+str(i+1)]
            A = self.relu(Z)
            self.caches.append(A)
        
        Z = np.dot(A_prev, self.parameters['W'+str(len(self.layers))]) + self.parameters['b'+str(len(self.layers))]
        AL = self.softmax(Z)
        self.caches.append(AL)
    
    # Cost function
    def compute_cost(self, AL, Y):

        loss = -1 * np.sum(np.dot(np.transpose(Y), np.log(AL)) + \
                           np.dot(np.transpose(1-Y), np.log(1-AL)))
        
        loss = loss / Y.shape[0]

        return loss
    
    # Backward Function
    def backward(self, X, Y):

        As = {}
        As['A'+str(0)] = X
        m = X.shape[0]

        for i in range(len(self.layers)):
            
            As['A'+str(i+1)] = self.caches[i]
        
        dZL = As['A'+str(len(self.layers))] - Y
        self.gradients['W'+str(len(self.layers))] = np.dot(dZL, As['A'+str(len(self.layers))].T) / m
        self.gradients['b'+str(len(self.layers))] = np.sum(dZL, axis=1, keepdims=True) / m

        dZ = dZL

        for i in reversed(range(len(self.layers - 1))):

            dZ_prev = dZ

            dZ = np.dot(np.dot(self.parameters['W'+str(i+2)], dZ_prev), [1 if As['A'+str(i+1)] > 0 else 0])
            self.gradients['W'+str(i+1)] = np.dot(dZ, As['A'+str(i)].T) / m
            self.gradients['b'+str(i+1)] = np.sum(dZ, axis=1, keepdims=True) / m
    
    # Update parameters
    def update_params(self, lr):

        for i in range(len(self.layers)):

            self.parameters['W'+str(i+1)] -= lr * self.gradients['W'+str(i+1)]
            self.parameters['b'+str(i+1)] -= lr * self.gradients['b'+str(i+1)]
    
    
    # Train
    def train(self, X, Y, epochs, lr):

        cost = []
        for e in range(epochs):

            self.forward(X)

            loss = self.compute_cost(self.caches[len(self.layers)], Y)
            if e % 100 == 0:
                cost.append(loss)
                print("Cost after {}th iteration: {:.3f}".format(e, loss))
            
            self.backward(X, Y)

            self.update_params(lr)

        return cost

def dictionary_to_vector(params_dict):

    flag = True

    for key in params_dict.keys():

        new_v = np.reshape(params_dict[key], (-1, 1))
        if flag:
            theta = new_v
        else:
            theta = np.concatenate((theta, new_v))
        flag = False
    
    return theta

def vector_to_dictionary(vector, layers):
    L = len(layers)
    parameters = {}
    k = 0

    for l in range(1, L):
        w_dim = layers[l] * layers[l - 1]
        b_dim = layers[l]

        temp_dim = k + w_dim

        parameters["W" + str(l)] = vector[
            k:temp_dim].reshape(layers[l], layers[l - 1])
        parameters["b" + str(l)] = vector[
            temp_dim:temp_dim + b_dim].reshape(b_dim, 1)

        k += w_dim + b_dim

    return parameters


def gradients_to_vector(gradients):
    valid_grads = [key for key in gradients.keys()
                   if not key.startswith("dA")]
    L = len(valid_grads)// 2
    flag = True
    
    for l in range(1, L + 1):
        if flag:
            new_grads = gradients["dW" + str(l)].reshape(-1, 1)
            new_grads = np.concatenate(
                (new_grads, gradients["db" + str(l)].reshape(-1, 1)))
        else:
            new_grads = np.concatenate(
                (new_grads, gradients["dW" + str(l)].reshape(-1, 1)))
            new_grads = np.concatenate(
                (new_grads, gradients["db" + str(l)].reshape(-1, 1)))
        flag = False
        
    return new_grads