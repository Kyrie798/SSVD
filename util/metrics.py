import numpy as np
from torch.nn.modules.loss import _Loss
from skimage.metrics import structural_similarity as compare_ssim

from util.utils import normalize_reverse

class PSNR(_Loss):
    def __init__(self, centralize=True, normalize=True, val_range=255.):
        super(PSNR, self).__init__()
        self.centralize = centralize
        self.normalize = normalize
        self.val_range = val_range

    def _quantize(self, img):
        img = normalize_reverse(img, centralize=self.centralize, normalize=self.normalize, val_range=self.val_range)
        img = img.clamp(0, self.val_range).round()
        return img

    def forward(self, x, y):
        diff = self._quantize(x) - self._quantize(y)
        if x.dim() == 3:
            n = 1
        elif x.dim() == 4:
            n = x.size(0)
        elif x.dim() == 5:
            n = x.size(0) * x.size(1)

        mse = diff.div(self.val_range).pow(2).view(n, -1).mean(dim=-1)
        psnr = -10 * mse.log10()

        return psnr.mean()
    
def psnr_calculate(x, y, val_range=255.0):
    x = x.astype(np.float)
    y = y.astype(np.float)
    diff = (x - y) / val_range
    mse = np.mean(diff ** 2)
    psnr = -10 * np.log10(mse)
    return psnr


def ssim_calculate(x, y, val_range=255.0):
    ssim = compare_ssim(y, x, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                        data_range=val_range)
    return ssim