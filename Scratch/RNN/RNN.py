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
        self.dWax = np.zeros((hidden_size, in_size))
        self.dWaa = np.zeros((hidden_size, hidden_size))
        self.dba = np.zeros((hidden_size, 1))
    
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
    
    def cell_backward(self, xt, a_prev, da_next):
        dz = da_next * (1 - np.square(np.tanh(np.dot(self.Wax, xt) + 
                         np.dot(self.Waa, a_prev) + 
                         self.ba)))
        
        dxt = np.dot(self.Wax.T, dz)
        dWax = np.dot(dz, xt.T)
        
        da_prev = np.dot(self.Waa.T, dz)
        dWaa = np.dot(dz, a_prev.T)
        
        db = np.sum(dz, axis=1, keepdims=True)
        
        return dxt, dWax, da_prev, dWaa, db
    
    def backward(self, da, a, x):
        n_x, m, T_x = x.shape
        
        dx = np.zeros((x.shape))
        da_prevt = np.zeros((self.hidden_size, m))
        
        for t in range(T_x):
            
            dxt, dWaxt, da_prevt, dWaat, dbt = self.cell_backward(
                x[:,:,t], a[:,:,t],
                da[:,:,t] + da_prevt
            )
            dx[:,:,t] = dxt
            self.dWax += dWaxt
            self.dWaa += dWaat
            self.dba += dbt