import numpy as np
from Utils import DeepNet

class SGD(DeepNet):
    
    def __init__(self):
        super(SGD, self).__init__()
    
    def train_with_sgd(self, X, Y, epochs, lr):
        m = X.shape[0]
        cost = []
        permutation = list(np.random.permutation(m))
        X = X[:, permutation]
        Y = Y[:, permutation]
        
        for e in range(epochs):
            for i in range(m):
                self.forward(X[i,:])

                loss = self.compute_cost(self.caches[len(self.layers)], Y[i,:])
                if e % 100 == 0:
                    cost.append(loss)
                
                self.backward(X[i,:], Y[i,:])

                self.update_params(lr)
            print("Cost after {}th iteration: {:.3f}".format(e, loss))

        return cost

class MiniBGD(DeepNet):
    
    def __init__(self, batch_size):
        super(MiniBGD, self).__init__()
        self.batch_size = batch_size
    
    def generate_mini_batches(self, X, Y):
        
        m = X.shape[0]
        mini_batches = []
        permutation = list(np.random.permutation(m))
        X = X[:, permutation]
        Y = Y[:, permutation]
        
        num_of_batches = np.floor(m / self.batch_size)
        for k in range(num_of_batches):
            
            mini_X = X[k * self.batch_size : (k+1) * self.batch_size, :]
            mini_Y = Y[k * self.batch_size : (k+1) * self.batch_size, :]
            mini_batch = (mini_X, mini_Y)
            mini_batches.append(mini_batch)
        
        if m % self.batch_size != 0:
            
            mini_X = X[num_of_batches*self.batch_size : m, :]
            mini_Y = Y[num_of_batches*self.batch_size : m, :]
            mini_batch = (mini_X, mini_Y)
            mini_batches.append(mini_batch)
            
            return mini_batches

    def train_with_minibgd(self, X, Y, epochs, lr):
        
        mini_batches = self.generate_mini_batches(X, Y)
        cost = []
        for e in range(epochs):
            for i in range(len(mini_batches)):
                X, Y = mini_batches[i]
                
                self.forward(X)

                loss = self.compute_cost(self.caches[len(self.layers)], Y)
                if e % 100 == 0:
                    cost.append(loss)
                
                self.backward(X, Y)

                self.update_params(lr)
            print("Cost after {}th iteration: {:.3f}".format(e, loss))
            
        return cost