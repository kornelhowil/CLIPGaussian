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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def img_denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).to("cuda")
    std = torch.tensor([0.229, 0.224, 0.225]).to("cuda")
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = image * std + mean
    return image


def img_normalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).to("cuda")
    std = torch.tensor([0.229, 0.224, 0.225]).to("cuda")
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def clip_normalize(image):
    image = F.interpolate(image, size=224, mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to("cuda")
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to("cuda")
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def load_image(img_path, img_height=None, img_width=None):
    image = Image.open(img_path)
    if img_width is not None:
        image = image.resize((img_width, img_height))  # change image size to (3, img_size, img_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = transform(image)[:3, :, :].unsqueeze(0)

    return image