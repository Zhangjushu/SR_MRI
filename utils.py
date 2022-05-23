import torch.nn as nn
import math
import torch
from copy import deepcopy
import os

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


def calc_psnr(img1, img2, max=255.0):
    return 10. * ((max ** 2) / ((img1 - img2) ** 2).mean()).log10()


def denormalize(img):
    img = img.mul(255.0).clamp(0.0, 255.0)
    return img


def para_state_dict(model, model_save_path):
    state_dict = deepcopy(model.state_dict())
    # model_save_path = os.path.join(model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        print("===> loading checkpoint: {}".format(model_save_path))
        loaded_paras = torch.load(model_save_path)
        for key in state_dict:  # 在新的网络模型中遍历对应参数
            if key in loaded_paras and state_dict[key].size() == loaded_paras[key].size():
                # print("成功初始化参数:", key)
                state_dict[key] = loaded_paras[key]
        print("成功初始化参数")
    return state_dict
