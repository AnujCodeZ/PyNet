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

class RMSProp(DeepNet):
    
    def __init__(self, beta=0.99):
        super(DeepNet, self).__init__()
        self.beta = beta
        self.averages = {}
        self.init_averages()
        
    def init_averages(self):
        
        for i in range(len(self.layers)):
            
            self.averages['W'+str(i+1)] = np.zeros(self.parameters['W'+str(i+1)].shape)
            self.averages['b'+str(i+1)] = np.zeros(self.parameters['b'+str(i+1)].shape)
    
    def update_params_with_rmsprop(self, lr=3e-3):
        
        for i in range(len(self.layers)):
            
            self.averages['W'+str(i+1)] = (np.multiply(self.beta, self.averages['W'+str(i+1)]) + 
                                           np.multiply((1 - self.beta), np.square(self.gradients['W'+str(i+1)])))
            self.averages['b'+str(i+1)] = (np.multiply(self.beta, self.averages['b'+str(i+1)]) + 
                                           np.multiply((1 - self.beta), np.square(self.gradients['b'+str(i+1))]))
            
            self.parameters['W'+str(i+1)] -= lr * np.divide(self.gradients['W'+str(i+1)],
                                                            np.sqrt(self.averages['W'+str(i+1)]))
            self.parameters['b'+str(i+1)] -= lr * np.divide(self.gradients['b'+str(i+1)],
                                                            np.sqrt(self.averages['b'+str(i+1)]))

class Adam(DeepNet):
    
    def __init__(self, beta1=0.9, beta2=0.99, epsilon=1e-9):
        super(DeepNet, self).__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.averages_v = {}
        self.averages_s = {}
        self.init_averages()
    
    def init_averages(self):
        
        for i in range(len(self.layers)):
            
            self.averages_v['W'+str(i+1)] = np.zeros(self.parameters['W'+str(i+1)].shape)
            self.averages_v['b'+str(i+1)] = np.zeros(self.parameters['b'+str(i+1)].shape)
            self.averages_s['W'+str(i+1)] = np.zeros(self.parameters['W'+str(i+1)].shape)
            self.averages_s['b'+str(i+1)] = np.zeros(self.parameters['b'+str(i+1)].shape)
            self.averages_v_c['W'+str(i+1)] = np.zeros(self.parameters['W'+str(i+1)].shape)
            self.averages_v_c['b'+str(i+1)] = np.zeros(self.parameters['b'+str(i+1)].shape)
            self.averages_s_c['W'+str(i+1)] = np.zeros(self.parameters['W'+str(i+1)].shape)
            self.averages_s_c['b'+str(i+1)] = np.zeros(self.parameters['b'+str(i+1)].shape)
    
    def update_params_with_adam(self, lr=3e-3, epoch):
        
        for i in range(len(self.layers)):
            
            self.averages_v['W'+str(i+1)] = (np.multiply(self.beta, self.averages_v['W'+str(i+1)]) + 
                                           np.multiply((1 - self.beta), self.gradients['W'+str(i+1)]))
            self.averages_v['b'+str(i+1)] = (np.multiply(self.beta, self.averages_v['b'+str(i+1)]) + 
                                           np.multiply((1 - self.beta), self.gradients['b'+str(i+1)]))
            self.averages_s['W'+str(i+1)] = (np.multiply(self.beta, self.averages_s['W'+str(i+1)]) + 
                                           np.multiply((1 - self.beta), np.square(self.gradients['W'+str(i+1)])))
            self.averages_s['b'+str(i+1)] = (np.multiply(self.beta, self.averages_s['b'+str(i+1)]) + 
                                           np.multiply((1 - self.beta), np.square(self.gradients['b'+str(i+1))]))
            
            self.averages_v_c['W'+str(i+1)] = np.divide(self.averages_v['W'+str(i+1)], 
                                                        (1 - np.power(self.beta1, epoch)))
            self.averages_v_c['b'+str(i+1)] = np.divide(self.averages_v['b'+str(i+1)], 
                                                        (1 - np.power(self.beta1, epoch)))
            self.averages_s_c['W'+str(i+1)] = np.divide(self.averages_v['W'+str(i+1)], 
                                                        (1 - np.power(self.beta2, epoch)))
            self.averages_v_c['b'+str(i+1)] = np.divide(self.averages_v['b'+str(i+1)], 
                                                        (1 - np.power(self.beta1, epoch)))

            self.parameters['W'+str(i+1)] -= lr * np.divide(self.averages_v_c['W'+str(i+1)],
                                                            np.sqrt(self.averages_s_c['W'+str(i+1)] + self.epsilon))
            self.parameters['b'+str(i+1)] -= lr * np.divide(self.averages_v_c['b'+str(i+1)],
                                                            np.sqrt(self.averages_s_c['b'+str(i+1)]+ self.epsilon))