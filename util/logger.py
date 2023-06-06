import os
import torch
import shutil
from datetime import datetime

class Logger():
    def __init__(self):
        now = datetime.now()
        now = now.strftime("%Y_%m_%d_%H_%M_%S")

        # 保存训练过程
        file_path = os.path.join("./experiment/", now, "log.txt")
        self.save_dir = os.path.dirname(file_path)
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger = open(file_path, "a+")
        self.register_dict = {}

    def __call__(self, *args, verbose=True, prefix="", timestamp=True):
        if timestamp:
            now = datetime.now()
            now = now.strftime("%Y/%m/%d, %H:%M:%S - ")
        else:
            now = ""
        info = prefix + now
        for msg in args:
            if not isinstance(msg, str):
                msg = str(msg)
            info += msg + "\n"
        self.logger.write(info)
        if verbose:
            print(info, end="")
        self.logger.flush()

    def __del__(self):
        self.logger.close()

    def register(self, name, epoch, value):
        if name in self.register_dict:
            self.register_dict[name][epoch] = value
            if value > self.register_dict[name]["max"]:
                self.register_dict[name]["max"] = value
            if value < self.register_dict[name]["min"]:
                self.register_dict[name]["min"] = value
        else:
            self.register_dict[name] = {}
            self.register_dict[name][epoch] = value
            self.register_dict[name]["max"] = value
            self.register_dict[name]["min"] = value

    def report(self, items, state, epoch):
        msg = "[{}] ".format(state.lower())
        state = "_" + state.lower()
        for i in range(len(items)):
            item, best = items[i]
            msg += "{} : {:.4f} (best {:.4f})".format(item, 
                                                      self.register_dict[item + state][epoch],
                                                      self.register_dict[item + state][best])
            if i < len(items) - 1:
                msg += ", "
        self(msg, timestamp=False)

    def is_best(self, epoch):
        item = self.register_dict["1*L1_Charbonnier_loss_color" + "_valid"]
        return item[epoch] == item["min"]

    def save(self, state, filename="checkpoint.pth.tar"):
        path = os.path.join(self.save_dir, filename)
        torch.save(state, path)
        if self.is_best(state["epoch"]):
            copy_path = os.path.join(self.save_dir, "model_best.pth.tar")
            shutil.copy(path, copy_path)