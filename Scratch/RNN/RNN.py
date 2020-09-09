import numpy as np

class RNN:
    
    def __init__(self, in_size, hidden_size, out_size):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.Wax = np.random.rand(hidden_size, in_size) * 0.01
        self.Waa = np.random.rand(hidden_size, hidden_size) * 0.01
        self.Wya = np.random.rand(out_size, hidden_size) * 0.01
        self.ba = np.zeros((hidden_size, 1))
        self.by = np.zeros((out_size, 1))
    
    def cell_forward(self, xt, a_prev):
        a_next = np.tanh(np.dot(self.Wax, xt) + 
                         np.dot(self.Waa, a_prev) + 
                         self.ba)
        yt_hat = softmax(np.dot(self.Wya, a_next) + self.by)
        return a_next, yt_hat
    
    def forward(self, x):
        n_x, m, T_x = x.shape
        a = np.zeros((self.hidden_size, m, T_x))
        y_hat = np.zeros((self.out_size, m, T_x))
        
        for t in range(T_x):
            xt = x[:,:,t]
            a_next, yt_hat = self.cell_forward(xt, a[:,:,t])
            a[:,:,t] = a_next
            y_hat[:,:,t] = yt_hat
        
        return a, y_hat
        