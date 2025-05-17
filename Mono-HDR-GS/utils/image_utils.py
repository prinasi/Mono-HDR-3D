#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from PIL import Image, ImageEnhance


scenes_factor = dict(
    bathroom=3.0,
    diningroom=3.5,
    desk=2.0,
    dog=3.5,
    sponza=1.5,
    bear=2.5,
    chair=1.3,
    sofa=2.0,
)


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def min_max_norm(tensor, normalize=True):
    # tensor range: [0, 1]
    # Conver to PIL Image and then np.array (output shape: (H, W))
    if normalize:
        img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return img
    return tensor


def brighten_images(image: str, factor: float = 1.5):
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy() * 255.0
        image = image.astype(np.uint8)
        image = Image.fromarray(image.transpose(1, 2, 0))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        raise TypeError(f"Unsupported type {type(image)}!")

    enhancer = ImageEnhance.Brightness(image=image)
    brightened = enhancer.enhance(factor=factor)

    image = torch.from_numpy(np.asarray(brightened)).float().cuda() / 255.0
    if image.shape[0] != 3:
        image = image.permute(2, 0, 1)
    return image


def luminance_loss(x, y, c1=0.01**2):
    """
    计算 SSIM 中的亮度损失。

    参数:
    - x: 预测图像，形状为 [C, H, W]
    - y: 真实图像，形状为 [C, H, W]
    - C1: 稳定常数，默认为 (0.01)^2

    返回:
    - 亮度损失，标量
    """
    # 计算每个通道的平均亮度
    mu_x = x.mean(dim=[1, 2], keepdim=True)
    mu_y = y.mean(dim=[1, 2], keepdim=True)

    # 计算亮度比较
    luminance = (2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)
    loss = 1 - luminance.mean()

    return loss