import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class L1_Charbonnier_loss_color(_Loss):
    def __init__(self):
        super().__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        diff_sq = diff * diff
        diff_sq_color = torch.mean(diff_sq, 1, True)
        error = torch.sqrt(diff_sq_color + self.eps * self.eps)
        loss = torch.mean(error)
        return loss


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = L1_Charbonnier_loss_color()

    def forward(self, x, y):
        if len(x.shape) == 5:
            b, n, c, h, w = x.shape
            x = x.reshape(b * n, c, h, w)
            y = y.reshape(b * n, c, h, w)
        loss_sub = self.losses(x, y)
        return {"L1_Charbonnier_loss_color": loss_sub, "all": loss_sub}