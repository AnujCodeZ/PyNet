import numpy as np


class Conv2D:

    def __init__(self, num_filters, kernel, stride=1, padding='VALID'):

        self.num_filters = num_filters
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def zero_pad(self, x, pad):

        x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad),
                           (0, 0)), mode='constant', constant_values=(0, 0))
        return x_pad

    def conv_single(self, x_slice, w_channel):

        z = np.multiply(x_slice, self.w[:, :, :, w_channel])
        z = np.sum(z)

        return z + self.b.squeeze()

    def init_params(self, x):

        self.w = np.random.randn(self.kernel, self.kernel,
                                 x.shape[2], self.num_filters)
        self.b = np.zeros((1, 1, 1, self.num_filters))

    def __call__(self, x):

        self.init_params()

        if self.padding == 'SAME':
            pad = (self.kernel - 1) / 2

        m = x.shape[0]
        n_w = int((x.shape[1] + 2 * pad - self.kernel) / self.stride) + 1
        n_h = int((x.shape[2] + 2 * pad - self.kernel) / self.stride) + 1
        out = np.zeros((x.shape[0], n_h, n_w, self.num_filters))
        
        
        x_pad = self.zero_pad(x, pad)
        
        for i in range(m):
            x_i = x_pad[i]
            for h in range(n_h):
                hs = h
                he = h + self.kernel
                for w in range(n_w):
                    ws = w
                    we = w + self.kernel
                    for c in range(self.num_filters):
                        x_slice = x_i[hs:he, vs:ve, :]
                        out[i, h, w, c] = self.conv_single(x_slice, c)
        
        return out
                                        