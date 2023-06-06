from model.ESTRNN import ESTRNN, feed, cost_profile
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.model = ESTRNN(opt)

    def forward(self, iter_samples):
        outputs = feed(self.model, iter_samples)
        return outputs

    def profile(self):
        H, W = 720, 1280
        seq_length = 2 + 2 + 1
        flops, params = cost_profile(self.model, H, W, seq_length)
        return flops, params
