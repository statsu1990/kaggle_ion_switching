import torch
import torch.nn as nn
import numpy as np

class MixUp(nn.Module):
    def __init__(self, alpha):
        super(MixUp, self).__init__()
        self.alpha = alpha

    def forward(self, x, y=None):
        if y is not None:
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).cuda()
            mix_rate = self.get_mixrate()

            mixed_x = mix_rate * x + (1 - mix_rate) * x[index,:]
            y_a = y
            y_b = y[index]

            return mixed_x, y_a, y_b, mix_rate
        else:
            return x

    def get_mixrate(self):
        if self.alpha > 0.:
            mixrate = np.random.beta(self.alpha, self.alpha)
            if mixrate < 0.5:
                mixrate = 1 - mixrate
        else:
            mixrate = 1.

        return mixrate


