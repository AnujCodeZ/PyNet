import numpy as np
from Utils import DeepNet

class GDWithMomentum(DeepNet):
    
    def __init__(self, beta=0.9):
        super(DeepNet, self).__init__()
        self.beta = beta
        self.averages = {}
        self.init_averages()
    
    def init_averages(self):
        
        for i in range(len(self.layers)):
            
            self.averages['W'+str(i+1)] = np.zeros(self.parameters['W'+str(i+1)].shape)
            self.averages['b'+str(i+1)] = np.zeros(self.parameters['b'+str(i+1)].shape)
    
    def update_params_with_momentum(self, lr=3e-3):
        
        for i in range(len(self.layers)):
            
            self.averages['W'+str(i+1)] = (np.multiply(self.beta, self.averages['W'+str(i+1)]) + 
                                           np.multiply((1 - self.beta), self.gradients['W'+str(i+1)]))
            self.averages['b'+str(i+1)] = (np.multiply(self.beta, self.averages['b'+str(i+1)]) + 
                                           np.multiply((1 - self.beta), self.gradients['b'+str(i+1)]))
            
            self.parameters['W'+str(i+1)] -= lr * self.averages['W'+str(i+1)]
            self.parameters['b'+str(i+1)] -= lr * self.averages['b'+str(i+1)]