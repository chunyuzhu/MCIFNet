#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:08:49 2024

@author: z
"""

import torch 
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
    
class Mamba_INR(nn.Module):
    def __init__(self, scale_ratio, n_select_bands,  n_bands,  feature_dim=64, mlp_dim=[256, 128]):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp_dim = mlp_dim
        self.encoder_HSI = nn.Sequential(
                  nn.Conv2d(n_bands, feature_dim, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.encoder_MSI = nn.Sequential(
                  nn.Conv2d(n_select_bands, feature_dim, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        
        self.model1 = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=feature_dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.model2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=feature_dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.model3 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=feature_dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.model4 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=feature_dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        imnet_in_dim = self.feature_dim + self.feature_dim + 2
        self.imnet = MLP(imnet_in_dim, out_dim=n_bands+1, hidden_list=self.mlp_dim)
        
        self.softmax = nn.Softmax()
    def query(self, feat, coord, hr_guide):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
                
                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                print('pred:',pred.shape)
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        return ret
    def forward(self, hsi, msi):
        
        B, c, H, W = msi.shape
        _, C, h, w,= hsi.shape
        # print('self.encoder_HSI(hsi):',self.encoder_HSI(hsi).shape)
        HSI_encode = self.encoder_HSI(hsi).reshape(B, -1, self.feature_dim )
        MSI_encode = self.encoder_MSI(msi).reshape(B, -1, self.feature_dim )
        
        # print('HSI_encode:',HSI_encode.shape)
        # print('MSI_encode:',MSI_encode.shape)
        
        HSI_Deep = self.model2(self.model1(HSI_encode)).reshape(B, self.feature_dim, h, w)
        MSI_Deep = self.model4(self.model3(MSI_encode)).reshape(B, self.feature_dim, H, W)
        # print('HSI_Deep:',HSI_Deep.shape)
        # print('MSI_Deep:',MSI_Deep.shape)
        
        coord = make_coord([H, W]).cuda()
        # inrF = self.query(spe, coord, spa).reshape(B, -1, H*W)  # BxCxHxW
        inrF = self.query(HSI_Deep, coord, MSI_Deep) # BxCxHxW
        
        return inrF, 0,0,0,0,0 
    


if __name__ == '__main__':
    scale_ratio = 4
    n_select_bands = 4
    n_bands = 103
    MSI = torch.randn(1, n_select_bands, 128, 128).cuda()
    HSI = torch.randn(1, n_bands, 32, 32).cuda()
    # Create an instance of the Vim model
    model = Mamba_INR(scale_ratio=4, n_select_bands=4,  n_bands=103, feature_dim=64, mlp_dim=[256, 128]).cuda()

    # Perform a forward pass through the model
    out = model(HSI,MSI)

    # Print the shape and output of the forward pass
    print(out[0].shape)
    # print(flop_count_table(FlopCountAnalysis(model, (rgb, ms))))
    # print(out)


    
    
