import numpy as np


class GRU:
    
    def __init__(self, in_size, hidden_size, out_size):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.Wcx = np.random.rand(hidden_size, in_size) * 0.01
        self.Wux = np.random.rand(hidden_size, in_size) * 0.01
        self.Wcc = np.random.rand(hidden_size, hidden_size) * 0.01
        self.Wuc = np.random.rand(hidden_size, hidden_size) * 0.01
        self.Wyc = np.random.rand(out_size, hidden_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))
        self.bu = np.zeros((hidden_size, 1))
        self.by = np.zeros((out_size, 1))
    
    def sigmoid(self, z):
        return 1. / (1. + np.exp(-z))
    
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))
    
    def cell_forward(self, xt, c_prev):
        c_t = np.tanh(np.dot(self.Wcc, c_prev) +
                      np.dot(self.Wcx, xt) + self.bc)
        u = sigmoid(np.dot(self.Wuc, c_prev) + 
                    np.dot(self.Wux, xt) + self.bu)
        c_next = u * c_t + (1 - u) * c_prev
        yt_hat = softmax(np.dot(self.Wyc, c_next) + self.by)
        
        return c_next, yt_hat
    
    def forward(self, x):
        n_x, m, T_x = x.shape
        c = np.zeros((self.hidden_size, m, T_x))
        y_hat = np.zeros((self.out_size, m, T_x))
        
        for t in range(T_x):
            xt = x[:,:,t]
            c_next, yt_hat = self.cell_forward(xt, c[:,:,t])
            c[:,:,t] = c_next
            y_hat[:,:,t] = yt_hat
        
        return c, y_hat