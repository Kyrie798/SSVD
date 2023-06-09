import os
import pickle
import random
import lmdb
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from util.utils import Crop, Flip, ToTensor, normalize

class DeblurDataset(Dataset):
    def __init__(self, opt, type):
        self.blur = os.path.join(opt.data_root, "gopro_ds_{}".format(type))
        self.gt = os.path.join(opt.data_root, "gopro_ds_{}_gt".format(type))
        with open(os.path.join(opt.data_root, "gopro_ds_info_{}.pkl".format(type)), "rb") as f:
            self.seqs_info = pickle.load(f)
        self.transform = transforms.Compose([Crop([256, 256]), Flip(), ToTensor()])
        self.frames = 8
        self.crop_h, self.crop_w = [256, 256]
        self.W = 960
        self.H = 540
        self.C = 3
        self.num_ff = 2
        self.num_pf = 2
        self.normalize = True
        self.centralize = True
        self.env_blur = lmdb.open(self.blur, map_size=10995116277)
        self.env_gt = lmdb.open(self.gt, map_size=10995116277)
        self.txn_blur = self.env_blur.begin()
        self.txn_gt = self.env_gt.begin()

    def __len__(self):
        return self.seqs_info["length"] - (self.frames - 1) * self.seqs_info["num"]

    def __getitem__(self, idx):
        idx +=1
        ori_idx = idx
        seq_idx, frame_idx = 0, 0
        blur_imgs, sharp_imgs = list(), list()
        for i in range(self.seqs_info["num"]):
            seq_length = self.seqs_info[i]["length"] - self.frames + 1
            if idx - seq_length <= 0:
                seq_idx = i
                frame_idx = idx - 1
                break
            else:
                idx -= seq_length

        top = random.randint(0, self.H - self.crop_h)
        left = random.randint(0, self.W - self.crop_w)
        flip_lr_flag = random.randint(0, 1)
        flip_ud_flag = random.randint(0, 1)
        sample = {"top": top, "left": left, "flip_lr": flip_lr_flag, "flip_ud": flip_ud_flag}

        for i in range(self.frames):
            try:
                blur_img, sharp_img = self.get_img(seq_idx, frame_idx + i, sample)
                blur_imgs.append(blur_img)
                sharp_imgs.append(sharp_img)
            except TypeError as err:
                print("Handling run-time error:", err)
                print("failed case: idx {}, seq_idx {}, frame_idx {}".format(ori_idx, seq_idx, frame_idx))
        blur_imgs = torch.cat(blur_imgs, dim=0)
        sharp_imgs = torch.cat(sharp_imgs[self.num_pf:self.frames - self.num_ff], dim=0)
        return blur_imgs, sharp_imgs
    
    def get_img(self, seq_idx, frame_idx, sample):
        code = "%03d_%08d" % (seq_idx, frame_idx)
        code = code.encode()
        blur_img = self.txn_blur.get(code)
        blur_img = np.frombuffer(blur_img, dtype="uint8")
        blur_img = blur_img.reshape(self.H, self.W, self.C)
        sharp_img = self.txn_gt.get(code)
        sharp_img = np.frombuffer(sharp_img, dtype="uint8")
        sharp_img = sharp_img.reshape(self.H, self.W, self.C)
        sample["image"] = blur_img
        sample["label"] = sharp_img
        sample = self.transform(sample)
        blur_img = normalize(sample["image"], centralize=self.centralize, normalize=self.normalize)
        sharp_img = normalize(sample["label"], centralize=self.centralize, normalize=self.normalize)

        return blur_img, sharp_img

    def __len__(self):
        return self.seqs_info["length"] - (self.frames - 1) * self.seqs_info["num"]