# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 20:58:02 2023

@author: zxc
"""


import torch
from torch import nn
import torch.nn.functional as F
# from torchvision.models.vgg import vgg16
import numpy as np
import cv2
# from scipy.fftpack import fft2
def set_grad(network,requires_grad):
        for param in network.parameters():
            param.requires_grad = requires_grad

def gaussian_pyramid(image_tensor, n_levels=3):
    # 创建一个空的图像列表来保存金字塔中的每一层
    pyramid = []

    # 将输入图像复制到列表中作为第一层
    pyramid.append(image_tensor)

    # 对于每一层（除了最后一层）
    for _ in range(n_levels - 1):
        # 计算当前层的高斯金字塔图像
        image_tensor = F.avg_pool2d(image_tensor, kernel_size=2, stride=2)
        
        pyramid.append(image_tensor)

    return pyramid

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        
        self.tv_loss = TVLoss()
        self.l1loss = nn.L1Loss()
        
    def forward(self, img_out_labels, out_1, out_2, out_3, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - img_out_labels)#+0.3*torch.mean(1 - edge_out_labels)+0.3*torch.mean(1 - spec_out_labels)
       
        target_gaussian = gaussian_pyramid(target_images)
        
       
        image_loss = self.l1loss(out_1, target_gaussian[0]) #+ self.l1loss(out_2, target_gaussian[1]) + self.l1loss(out_3, target_gaussian[2])
       
        # tv_loss = self.tv_loss(out_images)
        # return image_loss + adversarial_loss  + 2e-8 * tv_loss
        # return image_loss   + 2e-8 * tv_loss
        return image_loss + adversarial_loss  #+ 2e-8 * tv_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = Con_Edge_Spec_loss()
    A = torch.rand(1,144,128,128)
    B = torch.rand(1,144,128,128)
    print(g_loss(A,B))
