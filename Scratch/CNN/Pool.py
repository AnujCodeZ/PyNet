import numpy as np


class Pool:

    def __init__(self, kernel, stride, mode='max'):
        self.kernel = kernel
        self.stride = stride
        self.mode = mode

    def __call__(self, x):

        m = x.shape[0]
        n_h = int(1 + (x.shape[1] - self.kernel) / self.stride)
        n_w = int(1 + (x.shape[2] - self.kernel) / self.stride)
        n_c = x.shape[3]

        x_pool = np.zeros((m, n_h, n_w, n_c))
        for i in range(m):
            x_i = x[i]
            for h in range(n_h):
                for w in range(n_w):
                    for c in range(n_c):
                        hs = h
                        he = h + self.kernel
                        ws = w
                        we = w + self.kernel
                        x_slice = x_i[hs:he, ws:we, c]

                        if self.mode == 'max':
                            x_pool[i, h, w, c] = np.max(x_slice)
                        elif self.mode == 'average':
                            x_pool[i, h, w, c] = np.mean(x_slice)

        return x_pool
