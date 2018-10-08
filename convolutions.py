class ResidualGCNBuilder(object):
    def __init__(self, layers, width, height, channel, p_drop, model):
        pc = model.add_subcollection("ResidualGCNBuilder")
        self.filters = [pc.add_parameters((width, height, channel, 2 * channel))]
        for i in range(1, layers):
            self.filters.append(pc.add_parameters((width, height, channel, 2 * channel)))
        self.channel = channel
        self.p_drop = p_drop
        self.pc = pc
        self.spec = (layers, width, height, channel, p_drop)

    def __call__(self, x, train):
        for flt in self.filters:
            x_res = dy.conv2d(x, flt, stride=(1, 1), is_valid=False)
            x_res = dy.cmult(dy.logistic(x_res[:, :, :self.channel]), dy.dropout(x_res[:, :, self.channel:], self.p_drop if train else 0.0))
            x += x_res
        return x 
    
class GCNBuilder(object):
    def __init__(self, layers, width, height, channel, p_drop, model):
        pc = model.add_subcollection("GCNBuilder")
        self.filters = [pc.add_parameters((width, height, channel, 2 * channel))]
        for i in range(1, layers):
            self.filters.append(pc.add_parameters((width, height, channel, 2 * channel)))
        self.channel = channel
        self.p_drop = p_drop
        self.pc = pc
        self.spec = (layers, width, height, channel, p_drop)

    def __call__(self, x, train):
        for flt in self.filters:
            x_res = dy.conv2d(x, flt, stride=(1, 1), is_valid=False)
            x_res = dy.cmult(dy.logistic(x_res[:, :, :self.channel]), x_res[:, :, self.channel:])
            x = x_res
        return x
    

