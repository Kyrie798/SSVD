import os
import torch
import argparse
import random
import time
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from model.model import Model
from util.logger import Logger
from loss import Loss
from util.metrics import PSNR
from dataset import DeblurDataset
from util.utils import AverageMeter

class Trainer:
    def __init__(self, opt):
        self.opt = opt
    
    def train(self):
        # setup
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(20)
        torch.cuda.manual_seed(20)
        random.seed(20)
        np.random.seed(20)
        
        # logger
        logger = Logger()
        logger.writer = SummaryWriter(logger.save_dir)

        # create model
        model = Model(self.opt).cuda()

        # create criterion
        criterion = Loss().cuda()
        
        # create measurement according to metrics
        metrics = PSNR(centralize=True, normalize=True, val_range=2.0 ** 8 - 1)

        # create optimizer
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-8)
        lr = optimizer.param_groups[0]["lr"]

        # distributed data parallel
        model = nn.DataParallel(model)

        # create dataset&&dataloader
        train_dataset = DeblurDataset(opt, type="train")
        val_dataset = DeblurDataset(opt, type="valid")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        # resume from a checkpoint
        if opt.resume:
            if os.path.isfile(opt.resume_file):
                checkpoint = torch.load(opt.resume_file, map_location="cuda:0")
                logger("loading checkpoint {} ...".format(opt.resume_file))
                logger.register_dict = checkpoint["register_dict"]
                opt.start_epoch = checkpoint["epoch"] + 1
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
            else:
                msg = "no check point found at {}".format(opt.resume_file)
                logger(msg, verbose=False)
                raise FileNotFoundError(msg)

        for epoch in range(opt.start_epoch, opt.epochs + 1):
            # train
            model.train()
            logger("[Epoch [{}/{}] lr {:.2e}]".format(epoch, opt.epochs, lr), prefix="\n")
            losses_meter = {}
            losses_name = ["L1_Charbonnier_loss_color"]
            losses_name.append("all")
            for key in losses_name:
                losses_meter[key] = AverageMeter()

            measure_meter = AverageMeter()
            batchtime_meter = AverageMeter()

            start = time.time()
            end = time.time()
            pbar = tqdm(total=len(train_dataloader), ncols=80)
            for iter_samples in train_dataloader:
                for (key, val) in enumerate(iter_samples):
                    iter_samples[key] = val.cuda()
                inputs = iter_samples[0]
                labels = iter_samples[1]
                outputs = model(iter_samples)

                losses = criterion(outputs, labels)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                measure = metrics(outputs.detach(), labels)

                for key in losses_name:
                    losses_meter[key].update(losses[key].detach().item(), inputs.size(0))
                measure_meter.update(measure.detach().item(), inputs.size(0))

                optimizer.zero_grad()
                losses["all"].backward()

                clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

                optimizer.step()

                batchtime_meter.update(time.time() - end)
                end = time.time()
                pbar.update(1)
            pbar.close()
            # record info
            logger.register("1*L1_Charbonnier_loss_color" + "_train", epoch, losses_meter["all"].avg)
            logger.register("PSNR" + "_train", epoch, measure_meter.avg)
            for key in losses_name:
                logger.writer.add_scalar(key + "_loss_train", losses_meter[key].avg, epoch)
            logger.writer.add_scalar("PSNR" + "_train", measure_meter.avg, epoch)
            logger.writer.add_scalar("lr", lr, epoch)

            # show info
            logger("[train] epoch time: {:.2f}s, average batch time: {:.2f}s".format(end - start, batchtime_meter.avg),
                timestamp=False)
            logger.report([["1*L1_Charbonnier_loss_color", "min"], ["PSNR", "max"]], state="train", epoch=epoch)
            msg = "[train]"
            for key, meter in losses_meter.items():
                if key == "all":
                    continue
                msg += " {} : {:4f};".format(key, meter.avg)
            logger(msg, timestamp=False)
            scheduler.step()

            # val
            model.eval()
            with torch.no_grad():
                losses_meter = {}
                losses_name = ["L1_Charbonnier_loss_color"]
                losses_name.append("all")
                for key in losses_name:
                    losses_meter[key] = AverageMeter()

                measure_meter = AverageMeter()
                batchtime_meter = AverageMeter()
                start = time.time()
                end = time.time()
                pbar = tqdm(total=len(val_dataloader), ncols=80)

                for iter_samples in val_dataloader:
                    for (key, val) in enumerate(iter_samples):
                        iter_samples[key] = val.cuda()
                    inputs = iter_samples[0]
                    labels = iter_samples[1]
                    outputs = model(iter_samples)

                    losses = criterion(outputs, labels)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    measure = metrics(outputs.detach(), labels)
                    for key in losses_name:
                        losses_meter[key].update(losses[key].detach().item(), inputs.size(0))
                    measure_meter.update(measure.detach().item(), inputs.size(0))

                    batchtime_meter.update(time.time() - end)
                    end = time.time()
                    pbar.update(1)

            pbar.close()

            # record info
            logger.register("1*L1_Charbonnier_loss_color" + "_valid", epoch, losses_meter["all"].avg)
            logger.register("PSNR" + "_valid", epoch, measure_meter.avg)
            for key in losses_name:
                logger.writer.add_scalar(key + "_loss_valid", losses_meter[key].avg, epoch)
            logger.writer.add_scalar("PSNR" + "_valid", measure_meter.avg, epoch)

            # show info
            logger("[valid] epoch time: {:.2f}s, average batch time: {:.2f}s".format(end - start, batchtime_meter.avg),
                timestamp=False)
            logger.report([["1*L1_Charbonnier_loss_color", "min"], ["PSNR", "max"]], state="valid", epoch=epoch)
            msg = "[valid]"
            for key, meter in losses_meter.items():
                if key == "all":
                    continue
                msg += " {} : {:4f};".format(key, meter.avg)
            logger(msg, timestamp=False)

            # save checkpoint
            checkpoint = {"epoch": epoch,
                          "model": "ESTRNN",
                          "state_dict": model.state_dict(),
                          "register_dict": logger.register_dict,
                          "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict()}
            logger.save(checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_file", type=str)
    parser.add_argument("--data_root", type=str, default="./datasets/gopro_ds_lmdb")
    opt = parser.parse_args()
    trainer = Trainer(opt)
    trainer.train()