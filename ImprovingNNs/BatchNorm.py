import numpy as np
from Utils import DeepNet

class BatchNorm(DeepNet):
    
    def __init__(self):
        self.gammas = None
        self.betas = None
        
    def __call__(self, z, epsilon=1e-3):
        self.init_params(z)
        self.mu = np.mean(z, axis=0)
        self.sigma = np.var(z,axis=0)
        self.z_norm = (z - self.mu) / (np.sqrt(self.sigma + epsilon))
        self.z_tilda = np.dot(self.gammas, self.z_norm) + self.betas
        
        return self.z_tilda
    
    def init_params(self, z):
        self.gammas = np.random.randn(z.shape[0], z.shape[0])
        self.betas = np.zeros((1, z.shape[1]))
    
    def backward_with_batchnorm(self, dz_prev, layer_no):
        m = self.gammas.shape[0]
        dz = np.multiply(np.dot(self.gammas, dz_prev), [1 if self.caches[layer_no-1] > 0 else 0])
        self.grad_gammas = np.dot(dz, self.caches[layer_no - 2]) / self.gammas.shape[0]
        self.grad_betas = np.sum(dz, axis=1, keepdims=True)