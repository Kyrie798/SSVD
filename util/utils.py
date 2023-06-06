import torch
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Crop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        top, left = sample["top"], sample["left"]
        new_h, new_w = self.output_size
        sample["image"] = image[top: top + new_h,
                          left: left + new_w]
        sample["label"] = label[top: top + new_h,
                          left: left + new_w]

        return sample
    

class Flip(object):
    def __call__(self, sample):
        flag_lr = sample["flip_lr"]
        flag_ud = sample["flip_ud"]
        if flag_lr == 1:
            sample["image"] = np.fliplr(sample["image"])
            sample["label"] = np.fliplr(sample["label"])
        if flag_ud == 1:
            sample["image"] = np.flipud(sample["image"])
            sample["label"] = np.flipud(sample["label"])

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = np.ascontiguousarray(image.transpose((2, 0, 1))[np.newaxis, :])
        label = np.ascontiguousarray(label.transpose((2, 0, 1))[np.newaxis, :])
        sample["image"] = torch.from_numpy(image).float()
        sample["label"] = torch.from_numpy(label).float()
        return sample
    

def normalize(x, centralize=False, normalize=False, val_range=255.0):
    if centralize:
        x = x - val_range / 2
    if normalize:
        x = x / val_range

    return x


def normalize_reverse(x, centralize=False, normalize=False, val_range=255.0):
    if normalize:
        x = x * val_range
    if centralize:
        x = x + val_range / 2

    return x