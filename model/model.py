from model.ESTRNN import ESTRNN, feed
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.model = ESTRNN(opt)

    def forward(self, iter_samples):
        outputs = feed(self.model, iter_samples)
        return outputs
