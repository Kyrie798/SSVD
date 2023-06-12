import os
import torch
import pickle
import cv2
import torch.nn as nn
import lmdb
import time
import numpy as np
import argparse

from model.model import Model
from util.logger import Logger
from util.utils import AverageMeter, normalize, normalize_reverse
from util.metrics import psnr_calculate, ssim_calculate

class Tester:
    def __init__(self, opt):
        self.opt = opt
    
    def test(self):
        logger = Logger()
        model = Model(self.opt).cuda()
        checkpoint = torch.load("./experiment/2023_06_09_11_15_15/model_best.pth.tar", map_location="cuda:0")
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint["state_dict"])

        PSNR = AverageMeter()
        SSIM = AverageMeter()
        timer = AverageMeter()
        results_register = set()

        B, H, W, C = 1, 540, 960, 3
        blur = os.path.join(opt.data_root, "gopro_ds_valid")
        gt = os.path.join(opt.data_root, "gopro_ds_valid_gt")
        env_blur = lmdb.open(blur, map_size=100)
        env_gt = lmdb.open(gt, map_size=100)
        txn_blur = env_blur.begin()
        txn_gt = env_gt.begin()

        data_test_info_path = os.path.join(opt.data_root, "gopro_ds_info_valid.pkl")
        with open(data_test_info_path, "rb") as f:
            seqs_info = pickle.load(f)

        for seq_idx in range(seqs_info["num"]):
            seq_length = seqs_info[seq_idx]["length"]
            seq = "{:03d}".format(seq_idx)
            logger("seq {} image results generating ...".format(seq))
            save_dir = os.path.join(logger.save_dir, "resluts", seq)
            os.makedirs(save_dir, exist_ok=True)
            start = 0
            end = 20
            while (True):
                input_seq = []
                label_seq = []
                for frame_idx in range(start, end):
                    code = "%03d_%08d" % (seq_idx, frame_idx)
                    code = code.encode()
                    blur_img = txn_blur.get(code)
                    blur_img = np.frombuffer(blur_img, dtype="uint8")
                    blur_img = blur_img.reshape(H, W, C).transpose((2, 0, 1))[np.newaxis, :]
                    gt_img = txn_gt.get(code)
                    gt_img = np.frombuffer(gt_img, dtype="uint8")
                    gt_img = gt_img.reshape(H, W, C)
                    input_seq.append(blur_img)
                    label_seq.append(gt_img)
                input_seq = np.concatenate(input_seq)[np.newaxis, :]
                model.eval()
                with torch.no_grad():
                    input_seq = normalize(torch.from_numpy(input_seq).float().cuda(), centralize=True,
                                        normalize=True)
                    time_start = time.time()
                    output_seq = model([input_seq, ])
                    if isinstance(output_seq, (list, tuple)):
                        output_seq = output_seq[0]
                    output_seq = output_seq.squeeze(dim=0)
                    timer.update((time.time() - time_start) / len(output_seq), n=len(output_seq))
                for frame_idx in range(2, end - start - 2):
                    blur_img = input_seq.squeeze()[frame_idx]
                    blur_img = normalize_reverse(blur_img, centralize=True, normalize=True)
                    blur_img = blur_img.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
                    blur_img_path = os.path.join(save_dir, "{:08d}_input.png".format(frame_idx + start))
                    gt_img = label_seq[frame_idx]
                    gt_img_path = os.path.join(save_dir, "{:08d}_gt.png".format(frame_idx + start))
                    deblur_img = output_seq[frame_idx - 2]
                    deblur_img = normalize_reverse(deblur_img, centralize=True, normalize=True)
                    deblur_img = deblur_img.detach().cpu().numpy().transpose((1, 2, 0))
                    deblur_img = np.clip(deblur_img, 0, 255).astype(np.uint8)
                    deblur_img_path = os.path.join(save_dir, "{:08d}_result.png".format(frame_idx + start))
                    cv2.imwrite(blur_img_path, blur_img)
                    cv2.imwrite(gt_img_path, gt_img)
                    cv2.imwrite(deblur_img_path, deblur_img)
                    if deblur_img_path not in results_register:
                        results_register.add(deblur_img_path)
                        PSNR.update(psnr_calculate(deblur_img, gt_img))
                        SSIM.update(ssim_calculate(deblur_img, gt_img))
                if end == seq_length:
                    break
                else:
                    start = end - 2 - 2
                    end = start + 20
                    if end > seq_length:
                        end = seq_length
                        start = end - 20
        logger("Test images : {}".format(PSNR.count), prefix="\n")
        logger("Test PSNR : {}".format(PSNR.avg))
        logger("Test SSIM : {}".format(SSIM.avg))
        logger("Average time per image: {}".format(timer.avg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./datasets/gopro_ds_lmdb")
    opt = parser.parse_args()
    tester = Tester(opt)
    tester.test()