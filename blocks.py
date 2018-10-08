import numpy as np
import dynet as dy


class MLP(object):
    def __init__(self, pc, d_i, d_h, d_o):
        self.i2h = pc.add_parameters((d_h, d_i), init=dy.NormalInitializer())
        self.bh = pc.add_parameters((d_h,), init=dy.NormalInitializer())
        self.h2o = pc.add_parameters((d_o, d_h), init=dy.NormalInitializer())
        self.bo = pc.add_parameters((d_o,), init=dy.NormalInitializer())

    def __call__(self, x):
        hidden = dy.tanh(self.i2h * x + self.bh)
        out = dy.tanh(self.h2o * hidden + self.bo)
        return out

class BiLinear(object):
    def __init__(self, pc, dim, channels=1, bias_x=False, bias_y=False):
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.xdim = dim + int(bias_x)
        self.ydim = dim + int(bias_y)
        self.channels = channels
        if channels == 1:
            self.W = pc.add_parameters((self.ydim, self.xdim))
        else:
            self.W = pc.add_parameters((self.ydim, channels * self.xdim))
 
    def __call__(self, x, y):
        seq_len_x = x.dim()[0][-1]  # last_dim, aka seq_len
        seq_len_y = y.dim()[0][-1]
        if self.bias_x:
            x = dy.concatenate([x, dy.inputTensor(np.ones((1, seq_len_x), dtype=np.float32))])
        if self.bias_x:
            y = dy.concatenate([y, dy.inputTensor(np.ones((1, seq_len_y), dtype=np.float32))])
        if self.channels == 1:
            out = dy.transpose(y) * self.W * x
        else:
            yw = dy.transpose(y) * self.W
            yw = dy.reshape(yw, (seq_len_y * self.channels, self.xdim))
            ywx = yw * x
            out = dy.reshape(ywx, (seq_len_y, self.channels, seq_len_x))
        return out
